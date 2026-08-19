# Atlas AI Product Categorization — Vercel Deployment

A serverless FastAPI and modern Web interface for Hierarchical Product Categorization using ResNet-101 and Spatial Attention Constrained Beam Search.

---

## 📁 Directory Structure

```
vercel_deployment/
├── api/
│   ├── index.py              # FastAPI endpoints (/api/predict, /api/taxonomy, /api/health)
│   ├── engine.py             # Inference engine & attention generator
│   ├── models.py             # ResNet-101 encoder & LSTM-Attention decoder definitions
│   ├── word_map.json         # Token-to-index mapping
│   └── taxonomy_paths.json   # 52 Pre-compiled clothing category sequences
├── public/
│   ├── index.html            # Dark-mode glassmorphism Web UI
│   ├── style.css             # Responsive styling
│   └── app.js                # Frontend client logic & drag-and-drop
├── requirements.txt          # Python dependencies
├── vercel.json               # Vercel routing & serverless build config
└── README.md
```

---

## 🚀 Available API Endpoints

### 1. `POST /api/predict`
Predicts category from an uploaded image file (multipart/form-data).
* **Parameters**:
  * `file`: Binary image file (JPG / PNG / WEBP)
  * `beam_size`: Optional integer (1 to 10, default `5`)

**Response Example:**
```json
{
  "gender": "Men",
  "master_category": "Ethnic Wear",
  "sub_category": "Kurta",
  "taxonomy_path": ["Men", "Ethnic Wear", "Kurta"],
  "confidence_score": 0.9763,
  "log_prob_score": -0.0240,
  "attention_image_base64": "iVBORw0KGgoAAA..."
}
```

### 2. `POST /api/predict-url`
Predicts category directly from an image URL.
* **Body:**
```json
{
  "image_url": "https://example.com/product.jpg",
  "beam_size": 5
}
```

### 3. `GET /api/taxonomy`
Returns the full 52-category hierarchical taxonomy tree.

### 4. `GET /api/health`
Returns API and model runtime status.

---

## 💻 Local Testing

Run the FastAPI application with Uvicorn from within `vercel_deployment`:

```bash
cd "e:\Main Projects\product-categoriser\vercel_deployment"
python -m uvicorn api.index:app --reload --port 8000
```
Then visit:
* Web UI: `http://localhost:8000` (or `http://localhost:8000/public/index.html`)
* Interactive Swagger Docs: `http://localhost:8000/api/docs`

---

## ☁️ Deploy to Vercel

1. **Install Vercel CLI**:
   ```bash
   npm i -g vercel
   ```
2. **Deploy**:
   ```bash
   cd "e:\Main Projects\product-categoriser\vercel_deployment"
   vercel
   ```
3. Follow the CLI prompts to deploy directly to your Vercel account.
