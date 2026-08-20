import asyncio
import time
import uuid
from typing import Optional, Dict, Any
from PIL import Image
from fastapi import HTTPException
from api.engine import AtlasEngine

class InferenceTask:
    def __init__(self, image_pil: Image.Image, beam_size: int = 5, timeout: float = 30.0):
        self.task_id = str(uuid.uuid4())
        self.image_pil = image_pil
        self.beam_size = beam_size
        self.timeout = timeout
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None
        self.future: asyncio.Future = asyncio.get_running_loop().create_future()

class InferenceQueue:
    _instance: Optional['InferenceQueue'] = None

    def __init__(self, max_queue_size: int = 100):
        self.max_queue_size = max_queue_size
        self.queue: Optional[asyncio.Queue] = None
        self.worker_task: Optional[asyncio.Task] = None
        self.engine = AtlasEngine.get_instance()
        self._running = False

        # Metrics & Telemetry
        self.total_submitted = 0
        self.total_completed = 0
        self.total_failed = 0
        self.total_rejected = 0
        self.total_timed_out = 0
        self.avg_inference_time_ms = 0.0
        self.last_inference_time_ms = 0.0
        self.current_active_task: Optional[str] = None

    @classmethod
    def get_instance(cls, max_queue_size: int = 100) -> 'InferenceQueue':
        if cls._instance is None:
            cls._instance = InferenceQueue(max_queue_size=max_queue_size)
        return cls._instance

    def initialize_queue(self):
        if self.queue is None:
            self.queue = asyncio.Queue(maxsize=self.max_queue_size)

    async def start_worker(self):
        self.initialize_queue()
        if self.worker_task is None or self.worker_task.done():
            self._running = True
            self.worker_task = asyncio.create_task(self._worker_loop(), name='AtlasInferenceWorker')
            print(f'[QUEUE LOG] Inference Worker started. Max queue depth: {self.max_queue_size}')

    async def stop_worker(self):
        self._running = False
        if self.worker_task and not self.worker_task.done():
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        print('[QUEUE LOG] Inference Worker stopped.')

    async def _worker_loop(self):
        loop = asyncio.get_running_loop()
        while self._running:
            try:
                task: InferenceTask = await self.queue.get()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f'[QUEUE ERROR] Error getting item from queue: {e}')
                continue

            if task.future.done():
                self.queue.task_done()
                continue

            task.started_at = time.time()
            self.current_active_task = task.task_id

            try:
                result = await loop.run_in_executor(
                    None,
                    self.engine.predict_image,
                    task.image_pil,
                    task.beam_size
                )
                
                inference_ms = (time.time() - task.started_at) * 1000.0
                self.last_inference_time_ms = round(inference_ms, 2)
                if self.avg_inference_time_ms == 0.0:
                    self.avg_inference_time_ms = self.last_inference_time_ms
                else:
                    self.avg_inference_time_ms = round(0.85 * self.avg_inference_time_ms + 0.15 * self.last_inference_time_ms, 2)

                task.completed_at = time.time()
                self.total_completed += 1

                if not task.future.done():
                    task.future.set_result(result)
            except Exception as e:
                self.total_failed += 1
                if not task.future.done():
                    task.future.set_exception(e)
            finally:
                self.current_active_task = None
                self.queue.task_done()

    async def submit(self, image_pil: Image.Image, beam_size: int = 5, timeout: float = 30.0) -> Dict[str, Any]:
        self.initialize_queue()
        if not self._running or self.worker_task is None or self.worker_task.done():
            await self.start_worker()

        if self.queue.full():
            self.total_rejected += 1
            raise HTTPException(
                status_code=503,
                detail=f'Inference queue is at full capacity ({self.max_queue_size} pending tasks). Please try again shortly.'
            )

        task = InferenceTask(image_pil=image_pil, beam_size=beam_size, timeout=timeout)
        self.total_submitted += 1

        try:
            self.queue.put_nowait(task)
        except asyncio.QueueFull:
            self.total_rejected += 1
            raise HTTPException(
                status_code=503,
                detail='Inference queue is full.'
            )

        try:
            result = await asyncio.wait_for(task.future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            self.total_timed_out += 1
            raise HTTPException(
                status_code=504,
                detail=f'Inference request timed out after {timeout}s in queue.'
            )

    def get_status(self) -> Dict[str, Any]:
        qsize = self.queue.qsize() if self.queue else 0
        return {
            'queue_active': self._running,
            'queue_depth': qsize,
            'max_queue_capacity': self.max_queue_size,
            'active_task_id': self.current_active_task,
            'is_busy': self.current_active_task is not None,
            'total_submitted': self.total_submitted,
            'total_completed': self.total_completed,
            'total_failed': self.total_failed,
            'total_rejected': self.total_rejected,
            'total_timed_out': self.total_timed_out,
            'last_inference_time_ms': self.last_inference_time_ms,
            'avg_inference_time_ms': self.avg_inference_time_ms,
        }
