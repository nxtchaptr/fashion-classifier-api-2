// Sample image data URLs for quick testing
const SAMPLES = {
  kurta: "https://images.unsplash.com/photo-1583391733956-3750e0ff4e8b?w=400&q=80",
  saree: "https://images.unsplash.com/photo-1610030469983-98e550d6193c?w=400&q=80",
  hoodie: "https://images.unsplash.com/photo-1556905055-8f358a7a47b2?w=400&q=80",
  blouse: "https://images.unsplash.com/photo-1594633312681-425c7b97ccd1?w=400&q=80"
};

let currentTab = 'upload';
let currentSnippet = 'curl';
let latestJsonResponse = null;
let currentImageFile = null;
let taxonomyData = null;

document.addEventListener('DOMContentLoaded', () => {
  checkApiHealth();
  setupDropZone();
  setupClipboardPaste();
  updateSnippet();
});

// Switch Tab
function switchTab(tab) {
  currentTab = tab;
  document.getElementById('tabUpload').classList.toggle('active', tab === 'upload');
  document.getElementById('tabUrl').classList.toggle('active', tab === 'url');
  document.getElementById('uploadView').classList.toggle('active', tab === 'upload');
  document.getElementById('urlView').classList.toggle('active', tab === 'url');
}

// Health Check
async function checkApiHealth() {
  const pill = document.getElementById('statusPill');
  const text = document.getElementById('statusText');
  try {
    const res = await fetch('/api/health');
    if (res.ok) {
      const data = await res.json();
      text.textContent = data.model_loaded ? 'Model Ready (ResNet-101)' : 'API Ready (On-Demand)';
      pill.style.borderColor = 'rgba(16, 185, 129, 0.4)';
    } else {
      text.textContent = 'API Initializing';
    }
  } catch (err) {
    text.textContent = 'Standby Mode';
  }
}

// Setup Clipboard Paste
function setupClipboardPaste() {
  window.addEventListener('paste', (e) => {
    const items = (e.clipboardData || e.originalEvent.clipboardData).items;
    for (let index in items) {
      const item = items[index];
      if (item.kind === 'file' && item.type.startsWith('image/')) {
        const blob = item.getAsFile();
        switchTab('upload');
        handleFile(blob);
        break;
      }
    }
  });
}

// Drop Zone Setup
function setupDropZone() {
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');

  ['dragenter', 'dragover'].forEach(name => {
    dropZone.addEventListener(name, (e) => {
      e.preventDefault();
      dropZone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(name => {
    dropZone.addEventListener(name, (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
    });
  });

  dropZone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFile(files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleFile(e.target.files[0]);
    }
  });
}

function handleFile(file) {
  if (!file.type.startsWith('image/')) {
    alert('Please select a valid image file (JPG, PNG, WebP).');
    return;
  }
  currentImageFile = file;

  const reader = new FileReader();
  reader.onload = (e) => {
    document.getElementById('previewImg').src = e.target.result;
    classifyImageFile(file);
  };
  reader.readAsDataURL(file);
}

// Classify from File
async function classifyImageFile(file) {
  setLoading(true);
  const startTime = performance.now();
  const beamSize = document.getElementById('beamSize').value;

  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch(`/api/predict?beam_size=${beamSize}`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: 'Server Error' }));
      throw new Error(err.detail || 'Prediction failed');
    }

    const data = await response.json();
    const elapsed = Math.round(performance.now() - startTime);
    renderResults(data, elapsed);
    checkApiHealth();
  } catch (error) {
    alert(`Error: ${error.message}`);
    setLoading(false);
  }
}

// Classify from URL
async function predictFromUrl() {
  const url = document.getElementById('imageUrlInput').value.trim();
  if (!url) {
    alert('Please enter an image URL');
    return;
  }

  setLoading(true);
  document.getElementById('previewImg').src = url;
  const startTime = performance.now();
  const beamSize = document.getElementById('beamSize').value;

  try {
    const response = await fetch('/api/predict-url', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_url: url, beam_size: parseInt(beamSize) })
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: 'Server Error' }));
      throw new Error(err.detail || 'Prediction failed');
    }

    const data = await response.json();
    const elapsed = Math.round(performance.now() - startTime);
    renderResults(data, elapsed);
    checkApiHealth();
  } catch (error) {
    alert(`Error: ${error.message}`);
    setLoading(false);
  }
}

// Load Sample
function loadSample(key) {
  const url = SAMPLES[key];
  if (!url) return;
  document.getElementById('imageUrlInput').value = url;
  switchTab('url');
  predictFromUrl();
}

// Render Results
function renderResults(data, elapsedMs) {
  setLoading(false);
  latestJsonResponse = data;

  // Breadcrumbs
  const breadcrumbContainer = document.getElementById('taxonomyBreadcrumbs');
  breadcrumbContainer.innerHTML = '';
  (data.taxonomy_path || []).forEach((node, idx) => {
    if (idx > 0) {
      const sep = document.createElement('span');
      sep.className = 'crumb-sep';
      sep.textContent = '>';
      breadcrumbContainer.appendChild(sep);
    }
    const badge = document.createElement('span');
    badge.className = 'crumb-node';
    badge.textContent = node;
    breadcrumbContainer.appendChild(badge);
  });

  // Metrics
  const conf = data.confidence_score ? (data.confidence_score * 100).toFixed(1) : '0.0';
  document.getElementById('confScore').textContent = `${conf}%`;
  document.getElementById('genderBadge').textContent = data.gender || 'Unassigned';
  document.getElementById('subCatBadge').textContent = data.sub_category || 'General';
  document.getElementById('inferTime').textContent = `${elapsedMs}ms`;

  // Attention Map
  const attentionBox = document.getElementById('attentionBox');
  const attentionImg = document.getElementById('attentionImg');
  if (data.attention_image_base64) {
    attentionImg.src = `data:image/png;base64,${data.attention_image_base64}`;
    attentionBox.style.display = 'block';
  } else {
    attentionBox.style.display = 'none';
  }

  // JSON
  document.getElementById('jsonOutput').textContent = JSON.stringify(data, null, 2);
}

function setLoading(isLoading) {
  document.getElementById('emptyState').style.display = 'none';
  document.getElementById('loadingState').style.display = isLoading ? 'block' : 'none';
  document.getElementById('resultsContent').style.display = isLoading ? 'none' : 'block';
}

function copyJson() {
  if (!latestJsonResponse) return;
  navigator.clipboard.writeText(JSON.stringify(latestJsonResponse, null, 2));
  alert('JSON response copied to clipboard!');
}

// Snippet Generation
function switchSnippet(type) {
  currentSnippet = type;
  document.querySelectorAll('.snippet-tab').forEach(b => {
    b.classList.toggle('active', b.textContent.toLowerCase().includes(type));
  });
  updateSnippet();
}

function updateSnippet() {
  const origin = window.location.origin;
  const beamSize = document.getElementById('beamSize')?.value || 5;
  const codeEl = document.getElementById('codeSnippetText');
  if (!codeEl) return;

  if (currentSnippet === 'curl') {
    codeEl.textContent = `curl -X POST "${origin}/api/predict?beam_size=${beamSize}" \\
  -F "file=@product_image.jpg"`;
  } else if (currentSnippet === 'python') {
    codeEl.textContent = `import requests

url = "${origin}/api/predict"
params = {"beam_size": ${beamSize}}
files = {"file": open("product_image.jpg", "rb")}

response = requests.post(url, params=params, files=files)
data = response.json()
print("Taxonomy Path:", data["taxonomy_path"])
print("Confidence:", data["confidence_score"])`;
  } else if (currentSnippet === 'js') {
    codeEl.textContent = `const formData = new FormData();
formData.append("file", fileInputElement.files[0]);

const response = await fetch("${origin}/api/predict?beam_size=${beamSize}", {
  method: "POST",
  body: formData
});
const data = await response.json();
console.log(data);`;
  }
}

function copySnippet() {
  const code = document.getElementById('codeSnippetText').textContent;
  navigator.clipboard.writeText(code);
  alert('Integration code copied to clipboard!');
}

// Taxonomy Modal
async function openTaxonomyModal() {
  const modal = document.getElementById('taxonomyModal');
  const body = document.getElementById('taxonomyModalBody');
  modal.classList.add('active');

  if (!taxonomyData) {
    body.innerHTML = '<div class="spinner"></div><p style="text-align:center;color:#94a3b8;">Fetching complete taxonomy tree...</p>';
    try {
      const res = await fetch('/api/taxonomy');
      const data = await res.json();
      taxonomyData = data;
      renderTaxonomyTree(data);
    } catch (e) {
      body.innerHTML = `<p style="color:#ef4444;text-align:center;">Failed to load taxonomy: ${e.message}</p>`;
    }
  } else {
    renderTaxonomyTree(taxonomyData);
  }
}

function closeTaxonomyModal(e) {
  if (e && e.target !== e.currentTarget) return;
  document.getElementById('taxonomyModal').classList.remove('active');
}

function renderTaxonomyTree(data) {
  const body = document.getElementById('taxonomyModalBody');
  const tree = data.taxonomy_tree || {};

  let html = `<p class="modal-sub">Total Hierarchy Categories: <strong>${data.total_paths || 52} paths</strong></p><div class="tree-container">`;

  for (const gender in tree) {
    html += `<div class="tree-gender-group">
      <div class="tree-gender-title">👤 ${gender}</div>
      <div class="tree-master-list">`;
    for (const master in tree[gender]) {
      html += `<div class="tree-master-item">
        <span class="tree-master-badge">📂 ${master}</span>
        <div class="tree-sub-chips">`;
      const subCats = Object.keys(tree[gender][master]);
      if (subCats.length === 0) {
        html += `<span class="tree-sub-leaf">${master}</span>`;
      } else {
        subCats.forEach(sub => {
          html += `<span class="tree-sub-leaf">${sub}</span>`;
        });
      }
      html += `</div></div>`;
    }
    html += `</div></div>`;
  }
  html += `</div>`;
  body.innerHTML = html;
}

