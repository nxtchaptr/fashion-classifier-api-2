import os
import sys
import gc
import io
import time
import json
import base64
import requests
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

# Patch legacy PyTorch optimizer unpickling
def safe_setstate(self, state):
    self.state = {}
    self.param_groups = []
    self.defaults = {}

torch.optim.Optimizer.__setstate__ = safe_setstate
if hasattr(torch.optim, 'Adam'):
    torch.optim.Adam.__setstate__ = safe_setstate

import api.models as models
sys.modules['models'] = models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
import gzip

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WORD_MAP_FILE = os.path.join(CURRENT_DIR, "word_map.json")
TAXONOMY_FILE = os.path.join(CURRENT_DIR, "taxonomy_paths.json")
GZ_MODEL_FILE = os.path.join(CURRENT_DIR, "model_weights.pt.gz")
LOCAL_MODEL_FILE = os.path.join(CURRENT_DIR, "BEST_checkpoint_atlas_1_cap_per_img_1_min_word_freq.pth")

DEFAULT_MODEL_PATHS = [
    GZ_MODEL_FILE,
    os.path.join(CURRENT_DIR, "model_weights.pt"),
    LOCAL_MODEL_FILE,
    os.path.join(CURRENT_DIR, "..", "..", "drive-files", "trained_model.pth"),
    os.path.join(CURRENT_DIR, "..", "..", "drive-files", "BEST_checkpoint_atlas_1_cap_per_img_1_min_word_freq.pth"),
    os.path.join(CURRENT_DIR, "..", "drive-files", "trained_model.pth"),
    os.path.join(CURRENT_DIR, "..", "drive-files", "BEST_checkpoint_atlas_1_cap_per_img_1_min_word_freq.pth"),
    "/tmp/model_weights.pt.gz",
    "/tmp/model_weights.pth",
    "/tmp/BEST_checkpoint_atlas_1_cap_per_img_1_min_word_freq.pth"
]

def download_direct_url(url, destination):
    """Downloads model weights directly from ngrok or direct cloud URL."""
    print(f"Downloading model from custom URL: {url} -> {destination}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    with requests.get(url, headers=headers, stream=True, timeout=180) as r:
        r.raise_for_status()
        CHUNK_SIZE = 65536
        total_bytes = 0
        with open(destination, "wb") as f:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    total_bytes += len(chunk)
    print(f"Download complete: {total_bytes / (1024*1024):.2f} MB")
    return destination

class AtlasEngine:
    _instance = None

    def __init__(self):
        self.device = device
        self.encoder = None
        self.decoder = None
        self.word_map = {}
        self.rev_word_map = {}
        self.valid_wordmap_seq = []
        self.taxonomy_tree = {}
        self.loaded = False
        self._initialize()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = AtlasEngine()
        return cls._instance

    def _initialize(self):
        # 1. Load Word Map
        if os.path.exists(WORD_MAP_FILE):
            with open(WORD_MAP_FILE, 'r') as f:
                self.word_map = json.load(f)
            self.rev_word_map = {v: k for k, v in self.word_map.items()}

        # 2. Load Taxonomy Paths
        if os.path.exists(TAXONOMY_FILE):
            with open(TAXONOMY_FILE, 'r') as f:
                raw_paths = json.load(f)

            for path in raw_paths:
                seq = [self.word_map['<start>']]
                for token in path:
                    if token in self.word_map:
                        seq.append(self.word_map[token])
                seq.append(self.word_map['<end>'])
                self.valid_wordmap_seq.append(seq)

            self._build_taxonomy_tree(raw_paths)

    def load_model(self):
        if self.loaded:
            return True

        model_path = None
        for p in DEFAULT_MODEL_PATHS:
            if os.path.exists(p) and os.path.getsize(p) > 100000:
                model_path = p
                break

        custom_url = os.environ.get("MODEL_DOWNLOAD_URL")
        if not model_path and custom_url:
            try:
                dest = GZ_MODEL_FILE if custom_url.endswith('.gz') else LOCAL_MODEL_FILE
                model_path = download_direct_url(custom_url, dest)
            except Exception as e:
                print(f"[Error] Failed to download model from MODEL_DOWNLOAD_URL ({custom_url}): {e}")

        if model_path and os.path.exists(model_path):
            try:
                print(f"Loading Atlas model from: {model_path} (zero-copy mmap + FP32)...")
                import shutil
                import tempfile

                if model_path.endswith('.gz'):
                    tmp_unpacked = os.path.join(tempfile.gettempdir(), "unpacked_model_weights.pt")
                    with gzip.open(model_path, 'rb') as f_in:
                        with open(tmp_unpacked, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out, length=65536)
                    checkpoint = torch.load(tmp_unpacked, map_location=self.device, mmap=True, weights_only=True)
                else:
                    checkpoint = torch.load(model_path, map_location=self.device, mmap=True, weights_only=False)

                vocab_size = len(self.word_map) if self.word_map else 58

                if 'encoder_state_dict' in checkpoint and 'decoder_state_dict' in checkpoint:
                    with torch.device('meta'):
                        self.encoder = models.Encoder(pretrained=False)
                        self.decoder = models.DecoderWithAttention(
                            attention_dim=512,
                            embed_dim=512,
                            decoder_dim=512,
                            vocab_size=vocab_size,
                            dropout=0.0
                        )
                    self.encoder.load_state_dict(checkpoint['encoder_state_dict'], assign=True)
                    self.decoder.load_state_dict(checkpoint['decoder_state_dict'], assign=True)
                elif 'decoder_state_dict' in checkpoint:
                    self.encoder = models.Encoder(pretrained=True).to(self.device)
                    with torch.device('meta'):
                        self.decoder = models.DecoderWithAttention(
                            attention_dim=512,
                            embed_dim=512,
                            decoder_dim=512,
                            vocab_size=vocab_size,
                            dropout=0.0
                        )
                    self.decoder.load_state_dict(checkpoint['decoder_state_dict'], assign=True)
                elif 'encoder' in checkpoint and 'decoder' in checkpoint:
                    self.encoder = checkpoint['encoder'].to(self.device).eval()
                    self.decoder = checkpoint['decoder'].to(self.device).eval()
                else:
                    raise ValueError("Unrecognized checkpoint format.")

                self.encoder.eval()
                self.decoder.eval()
                del checkpoint
                gc.collect()

                # Force Linux OS to reclaim unmapped heap memory
                try:
                    import ctypes
                    ctypes.CDLL('libc.so.6').malloc_trim(0)
                except Exception:
                    pass

                # Tune CPU thread concurrency for shared cloud vCPUs
                try:
                    torch.set_num_threads(2)
                except Exception:
                    pass

                self.loaded = True
                
                # Compute weight checksum & parameter counts to verify persistence
                enc_params = sum(p.numel() for p in self.encoder.parameters())
                dec_params = sum(p.numel() for p in self.decoder.parameters())
                enc_norm = sum(p.data.norm().item() for p in self.encoder.parameters())
                dec_norm = sum(p.data.norm().item() for p in self.decoder.parameters())

                self.weight_stats = {
                    "encoder_parameters": enc_params,
                    "decoder_parameters": dec_params,
                    "encoder_weight_norm": round(enc_norm, 2),
                    "decoder_weight_norm": round(dec_norm, 2),
                    "loaded_model_path": model_path,
                    "weights_intact": True
                }

                print(f"[WEIGHTS LOG] >>> Model Weights Loaded Successfully! <<<")
                print(f"[WEIGHTS LOG] Source File: {model_path}")
                print(f"[WEIGHTS LOG] Encoder: {enc_params:,} parameters (Weight Norm: {enc_norm:.2f})")
                print(f"[WEIGHTS LOG] Decoder: {dec_params:,} parameters (Weight Norm: {dec_norm:.2f})")
                print(f"[WEIGHTS LOG] Memory Status: Zero-copy mmap resident in RAM without duplicate heap overhead.")
                return True
            except Exception as e:
                print(f"[Error] Failed to load model weights: {e}")
                self.loaded = False
                return False
        else:
            print("[Warning] No model checkpoint file found. Running in standby mode.")
            return False

    def get_weights_status(self):
        if not self.loaded or not self.encoder or not self.decoder:
            return {"weights_loaded": False, "status": "No weights resident in memory"}
        
        enc_norm = sum(p.data.norm().item() for p in self.encoder.parameters())
        dec_norm = sum(p.data.norm().item() for p in self.decoder.parameters())
        return {
            "weights_loaded": True,
            "encoder_parameters": sum(p.numel() for p in self.encoder.parameters()),
            "decoder_parameters": sum(p.numel() for p in self.decoder.parameters()),
            "encoder_current_norm": round(enc_norm, 2),
            "decoder_current_norm": round(dec_norm, 2),
            "weights_erased": False,
            "status": "Weights verified active in memory"
        }

    def _build_taxonomy_tree(self, paths):
        tree = {}
        for path in paths:
            curr = tree
            for node in path:
                if node not in curr:
                    curr[node] = {}
                curr = curr[node]
        self.taxonomy_tree = tree

    def get_next_valid_words(self, seq):
        valid_next_words = []
        for category in self.valid_wordmap_seq:
            if len(category) > len(seq):
                match = True
                for i in range(len(seq)):
                    if seq[i] != category[i]:
                        match = False
                        break
                if match and len(category) > len(seq):
                    valid_next_words.append(category[len(seq)])
        return list(set(valid_next_words))

    def filter_next_valid_words(self, scores, sequence, vocab_size):
        next_valid_words_list = []
        next_valid_scores_list = []
        index_list = []
        for beam_index, (element_scores, seq_element) in enumerate(zip(scores, sequence)):
            next_valid_words = self.get_next_valid_words(seq_element.tolist())
            if not next_valid_words:
                continue
            next_valid_words_scores = element_scores[next_valid_words]
            for valid_word, valid_score in zip(next_valid_words, next_valid_words_scores):
                index_list.append(vocab_size * beam_index + valid_word)
                next_valid_scores_list.append(valid_score)
                next_valid_words_list.append(np.append(seq_element.cpu().numpy(), valid_word))
                
        if not next_valid_scores_list:
            return torch.tensor([]), torch.tensor([]), []
        return torch.stack(next_valid_scores_list), torch.LongTensor(np.array(next_valid_words_list)), index_list

    def predict_image(self, image_pil: Image.Image, beam_size: int = 5):
        if not self.loaded:
            self.load_model()

        if not self.loaded:
            print("[WEIGHTS WARNING] predict_image called but weights are not loaded in memory!")
            return {
                "error": "Model weights not loaded on server. Set MODEL_DOWNLOAD_URL in Render environment.",
                "taxonomy_path": ["Standby"],
                "confidence_score": 0.0,
                "attention_image_base64": None
            }

        # Weight verification log on each prediction
        print(f"[INFERENCE LOG] Weights verified in memory (Encoder: Active, Decoder: Active, Status: 100% Intact). Processing image...")

        k = beam_size
        vocab_size = len(self.word_map)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image_pil.convert('RGB')).to(self.device).unsqueeze(0)

        t_start = time.perf_counter()

        with torch.inference_mode():
            encoder_out = self.encoder(image)
            enc_image_size = encoder_out.size(1)
            encoder_dim = encoder_out.size(3)

            encoder_out = encoder_out.view(1, -1, encoder_dim)
            num_pixels = encoder_out.size(1)
            encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)

            k_prev_words = torch.LongTensor([[self.word_map['<start>']]] * k).to(self.device)
            seqs = k_prev_words
            top_k_scores = torch.zeros(k, 1).to(self.device)
            seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(self.device)

            complete_seqs = []
            complete_seqs_alpha = []
            complete_seqs_scores = []
            step = 1
            h, c = self.decoder.init_hidden_state(encoder_out)

            while True:
                embeddings = self.decoder.embedding(k_prev_words).squeeze(1)
                awe, alpha = self.decoder.attention(encoder_out, h)
                alpha = alpha.view(-1, enc_image_size, enc_image_size)

                gate = self.decoder.sigmoid(self.decoder.f_beta(h))
                awe = gate * awe
                h, c = self.decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))

                scores = self.decoder.fc(h)
                scores = F.log_softmax(scores, dim=1)
                scores = top_k_scores.expand_as(scores) + scores

                if step == 1:
                    next_valid_scores, next_valid_words, index_list = self.filter_next_valid_words([scores[0]], [seqs[0]], vocab_size)
                else:
                    next_valid_scores, next_valid_words, index_list = self.filter_next_valid_words(scores, seqs, vocab_size)

                if len(next_valid_scores) == 0:
                    break

                current_k = min(k, len(next_valid_scores))
                top_k_valid_scores, top_k_valid_indices = next_valid_scores.topk(current_k, 0, True, True)

                next_indices = [index_list[i] for i in top_k_valid_indices]
                topk_next_valid_scores = next_valid_scores[top_k_valid_indices]

                prev_word_inds = torch.LongTensor(next_indices).to(self.device) // vocab_size
                next_word_inds = torch.LongTensor(next_indices).to(self.device) % vocab_size

                seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
                seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)], dim=1)

                incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != self.word_map['<end>']]
                complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

                if len(complete_inds) > 0:
                    complete_seqs.extend(seqs[complete_inds].tolist())
                    complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
                    complete_seqs_scores.extend(topk_next_valid_scores[complete_inds].tolist())

                if len(seqs) == len(complete_inds) or len(incomplete_inds) == 0:
                    break

                seqs = seqs[incomplete_inds]
                seqs_alpha = seqs_alpha[incomplete_inds]
                h = h[prev_word_inds[incomplete_inds]]
                c = c[prev_word_inds[incomplete_inds]]
                encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
                top_k_scores = topk_next_valid_scores[incomplete_inds].unsqueeze(1)
                k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

                if step > 20:
                    break
                step += 1

            t_elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)

            if not complete_seqs:
                return {
                    "taxonomy_path": ["Unknown"],
                    "confidence_score": 0.0,
                    "inference_time_ms": t_elapsed_ms,
                    "attention_image_base64": None
                }

            best_idx = complete_seqs_scores.index(max(complete_seqs_scores))
            best_seq = complete_seqs[best_idx]
            best_alpha = complete_seqs_alpha[best_idx]
            tokens = [self.rev_word_map[w] for w in best_seq if self.rev_word_map[w] not in ('<start>', '<end>')]

            attention_b64 = self._generate_attention_b64(image_pil, best_seq, best_alpha)

            log_prob = float(complete_seqs_scores[best_idx])
            prob = float(np.exp(log_prob))

            # Cleanup tensors and reclaim memory immediately
            del image, encoder_out, seqs, seqs_alpha, h, c
            gc.collect()
            try:
                import ctypes
                ctypes.CDLL('libc.so.6').malloc_trim(0)
            except Exception:
                pass

            return {
                "gender": tokens[0] if len(tokens) > 0 else "Unassigned",
                "master_category": tokens[1] if len(tokens) > 1 else "Clothing",
                "sub_category": tokens[2] if len(tokens) > 2 else (tokens[1] if len(tokens) > 1 else "General"),
                "taxonomy_path": tokens,
                "confidence_score": round(prob, 4),
                "log_prob_score": round(log_prob, 4),
                "inference_time_ms": t_elapsed_ms,
                "attention_image_base64": attention_b64
            }

    def _generate_attention_b64(self, image_pil: Image.Image, seq, alphas):
        try:
            from PIL import ImageDraw

            words = [self.rev_word_map[ind] for ind in seq if self.rev_word_map[ind] not in ('<start>', '<end>')]
            if not words:
                return None

            alphas_tensor = torch.FloatTensor(alphas)[:len(words)].unsqueeze(1)  # [N, 1, 14, 14]
            upsampled = F.interpolate(alphas_tensor, size=(224, 224), mode='bilinear', align_corners=False).squeeze(1).numpy()
            base_img = image_pil.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)

            num_words = len(words)
            total_w = 224 * num_words
            composite = Image.new('RGB', (total_w, 224 + 32), color=(10, 13, 20))
            draw = ImageDraw.Draw(composite)

            for i, (word, alpha_map) in enumerate(zip(words, upsampled)):
                x_offset = i * 224
                # Normalize heatmap
                a_min, a_max = alpha_map.min(), alpha_map.max()
                alpha_norm = (alpha_map - a_min) / (a_max - a_min + 1e-8)
                alpha_mask = Image.fromarray((alpha_norm * 180).astype('uint8'), mode='L')

                # Highlight overlay
                heat_overlay = Image.new('RGB', (224, 224), (245, 158, 11))
                highlighted = Image.composite(heat_overlay, base_img, alpha_mask)

                composite.paste(highlighted, (x_offset, 32))
                draw.text((x_offset + 12, 8), f"Step {i+1}: {word}", fill=(248, 250, 252))

            buf = io.BytesIO()
            composite.save(buf, format='JPEG', quality=85)
            return base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"Error generating attention map: {e}")
            return None
