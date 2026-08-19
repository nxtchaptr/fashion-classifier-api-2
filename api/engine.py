import os
import sys
import io
import re
import json
import base64
import requests
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform

# Patch legacy PyTorch optimizer unpickling
def safe_setstate(self, state):
    if not hasattr(self, 'defaults'):
        self.defaults = {}
    if isinstance(state, dict):
        for k, v in state.items():
            setattr(self, k, v)

torch.optim.Optimizer.__setstate__ = safe_setstate
torch.optim.Adam.__setstate__ = safe_setstate

import api.models as models
# Map 'models' namespace in sys.modules so unpickler resolves checkpoint classes
sys.modules['models'] = models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WORD_MAP_FILE = os.path.join(CURRENT_DIR, "word_map.json")
TAXONOMY_FILE = os.path.join(CURRENT_DIR, "taxonomy_paths.json")
LOCAL_MODEL_FILE = os.path.join(CURRENT_DIR, "BEST_checkpoint_atlas_1_cap_per_img_1_min_word_freq.pth")

# Google Drive File ID for model weights
GDRIVE_MODEL_FILE_ID = "1UhCzuSj9fqBCdKKai62_zHEKZnxxsPSo"

DEFAULT_MODEL_PATHS = [
    LOCAL_MODEL_FILE,
    os.path.join(CURRENT_DIR, "..", "..", "drive-files", "BEST_checkpoint_atlas_1_cap_per_img_1_min_word_freq.pth"),
    os.path.join(CURRENT_DIR, "..", "drive-files", "BEST_checkpoint_atlas_1_cap_per_img_1_min_word_freq.pth"),
    os.path.join(os.getcwd(), "drive-files", "BEST_checkpoint_atlas_1_cap_per_img_1_min_word_freq.pth"),
    "/tmp/BEST_checkpoint_atlas_1_cap_per_img_1_min_word_freq.pth"
]

def download_from_gdrive(file_id, destination):
    """Downloads model weights directly from Google Drive if not present on server."""
    print(f"Downloading model checkpoint from Google Drive (ID: {file_id}) to {destination}...")
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break
            
    if not token and response.text and "download_warning" in response.text:
        match = re.search(r'confirm=([0-9A-Za-z_]+)', response.text)
        if match:
            token = match.group(1)
            
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    elif "uc-download-link" in response.text:
        match = re.search(r'href="(/uc\?export=download[^"]+)"', response.text)
        if match:
            confirm_url = "https://docs.google.com" + match.group(1).replace("&amp;", "&")
            response = session.get(confirm_url, stream=True)

    CHUNK_SIZE = 65536
    total_bytes = 0
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                total_bytes += len(chunk)
                
    print(f"Downloaded model checkpoint: {total_bytes / (1024*1024):.2f} MB")
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

        # 3. Locate or Auto-Download Model Checkpoint
        model_path = None
        for p in DEFAULT_MODEL_PATHS:
            if os.path.exists(p) and os.path.getsize(p) > 1000000:
                model_path = p
                break

        if not model_path:
            # Auto-download on Render startup
            try:
                model_path = download_from_gdrive(GDRIVE_MODEL_FILE_ID, LOCAL_MODEL_FILE)
            except Exception as e:
                print(f"[Error] Failed to auto-download model from Google Drive: {e}")

        if model_path and os.path.exists(model_path):
            print(f"Loading Atlas model from: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.decoder = checkpoint['decoder'].to(self.device).eval()
            self.encoder = checkpoint['encoder'].to(self.device).eval()
            self.loaded = True
            print("Atlas Engine loaded successfully!")
        else:
            print("[Warning] No model checkpoint file found. Running in standby mode.")

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
            return {
                "error": "Model weights not loaded on server",
                "taxonomy_path": ["Mock", "Sample", "Item"],
                "confidence_score": 0.0,
                "attention_image_base64": None
            }

        k = beam_size
        vocab_size = len(self.word_map)

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image_pil.convert('RGB')).to(self.device).unsqueeze(0)

        with torch.no_grad():
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

            if not complete_seqs:
                return {
                    "taxonomy_path": ["Unknown"],
                    "confidence_score": 0.0,
                    "attention_image_base64": None
                }

            best_idx = complete_seqs_scores.index(max(complete_seqs_scores))
            best_seq = complete_seqs[best_idx]
            best_alpha = complete_seqs_alpha[best_idx]
            tokens = [self.rev_word_map[w] for w in best_seq if self.rev_word_map[w] not in ('<start>', '<end>')]

            attention_b64 = self._generate_attention_b64(image_pil, best_seq, best_alpha)

            log_prob = float(complete_seqs_scores[best_idx])
            prob = float(np.exp(log_prob))

            return {
                "gender": tokens[0] if len(tokens) > 0 else "Unassigned",
                "master_category": tokens[1] if len(tokens) > 1 else "Clothing",
                "sub_category": tokens[2] if len(tokens) > 2 else (tokens[1] if len(tokens) > 1 else "General"),
                "taxonomy_path": tokens,
                "confidence_score": round(prob, 4),
                "log_prob_score": round(log_prob, 4),
                "attention_image_base64": attention_b64
            }

    def _generate_attention_b64(self, image_pil: Image.Image, seq, alphas):
        try:
            image = image_pil.convert('RGB').resize([14 * 24, 14 * 24], Image.LANCZOS)
            words = [self.rev_word_map[ind] for ind in seq]
            alphas_tensor = torch.FloatTensor(alphas)

            num_words = len(words)
            cols = min(5, num_words)
            rows = int(np.ceil(num_words / float(cols)))
            fig = plt.figure(figsize=(cols * 2.8, rows * 2.8))

            for t in range(num_words):
                ax = fig.add_subplot(rows, cols, t + 1)
                ax.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=10, weight='bold')
                ax.imshow(image)
                current_alpha = alphas_tensor[t, :]
                alpha_img = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
                if t == 0:
                    ax.imshow(alpha_img, alpha=0)
                else:
                    ax.imshow(alpha_img, alpha=0.6, cmap=cm.Greys_r)
                ax.axis('off')

            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
            plt.close(fig)
            buf.seek(0)
            return base64.b64encode(buf.read()).decode('utf-8')
        except Exception as e:
            print(f"Error generating attention map: {e}")
            return None
