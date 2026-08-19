// Sample image data URLs for quick testing
const SAMPLES = {
  kurta: "https://images.unsplash.com/photo-1583391733956-3750e0ff4e8b?w=400&q=80",
  saree: "https://images.unsplash.com/photo-1610030469983-98e550d6193c?w=400&q=80",
  hoodie: "https://images.unsplash.com/photo-1556905055-8f358a7a47b2?w=400&q=80",
  blouse: "https://images.unsplash.com/photo-1594633312681-425c7b97ccd1?w=400&q=80"
};

let currentTab = 'upload';
let latestJsonResponse = null;

document.addEventListener('DOMContentLoaded', () => {
  checkApiHealth();
  setupDropZone();
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
      text.textContent = data.model_loaded ? 'Model Ready (ResNet-101)' : 'API Connected';
      pill.style.borderColor = 'rgba(16, 185, 129, 0.4)';
    } else {
      text.textContent = 'API Initializing';
    }
  } catch (err) {
    text.textContent = 'Local Standby';
  }
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
      const err = await response.json();
      throw new Error(err.detail || 'Prediction failed');
    }

    const data = await response.json();
    const elapsed = Math.round(performance.now() - startTime);
    renderResults(data, elapsed);
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
      const err = await response.json();
      throw new Error(err.detail || 'Prediction failed');
    }

    const data = await response.json();
    const elapsed = Math.round(performance.now() - startTime);
    renderResults(data, elapsed);
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
  document.getElementById('confScore').textContent = `${(data.confidence_score * 100).toFixed(1)}%`;
  document.getElementById('genderBadge').textContent = data.gender || 'Unassigned';
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
  alert('JSON copied to clipboard!');
}
