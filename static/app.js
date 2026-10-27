const $ = (id) => document.getElementById(id);

const drop = $("drop");
const gal = $("gal");
const box = $("box");
const preview = $("preview");
const out = $("out");
const cameraPanel = $("cameraPanel");
const cameraVideo = $("cameraVideo");
const cameraCanvas = $("cameraCanvas");
let currentFile = null;
let cameraStream = null;
let currentClassifier = "emotion";

function showError(message) {
  out.hidden = false;
  out.innerHTML = `<p class="error">${escapeHtml(message)}</p>`;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[ch]));
}

function formatPercent(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function renderResult(data) {
  out.hidden = false;
  const result = data.results && data.results[0];
  if (!result) {
    out.innerHTML = `<h2>UNABLE TO IDENTIFY</h2>
      <p>${escapeHtml(data.message || "Choose a classifier and try again.")}</p>
      <p class="status-label status-rejected">STATUS: SENT TO ROOM 101. Prediction did not meet 60% confidence threshold.</p>`;
    return;
  }

  const rejected = result.status === "LOW_CONFIDENCE" ||
    result.status === "SENT TO ROOM 101" ||
    Number(result.confidence) < Number(result.threshold ?? 0.60);
  const title = rejected ? "UNABLE TO IDENTIFY" : result.prediction;
  const statusText = rejected
    ? "STATUS: SENT TO ROOM 101. Prediction did not meet 60% confidence threshold."
    : "STATUS: Prediction met the 60% confidence threshold.";
  const entries = Object.entries(result.probabilities || {})
    .sort((a, b) => b[1] - a[1]);
  const bars = entries.map(([label, score]) => `
    <div class="probability-row">
      <span class="probability-label">${escapeHtml(label)}</span>
      <div class="probability-track"><div class="probability-fill" style="width:${Math.max(0, Math.min(100, Number(score) * 100))}%"></div></div>
      <span class="probability-value">${formatPercent(score)}</span>
    </div>`).join("");

  out.innerHTML = `
    <div class="classification-record">
      <div class="eyebrow">[ CLASSIFICATION RECORD ] <strong>${escapeHtml(result.model || "IMAGE")}</strong></div>
      <h2>${escapeHtml(title)}</h2>
      <p class="confidence-label">CONFIDENCE: ${formatPercent(result.confidence)}</p>
      <p class="status-label ${rejected ? "status-rejected" : "status-accepted"}">${escapeHtml(statusText)}</p>
      <div class="probabilities">${bars}</div>
    </div>`;
}

async function classify(file) {
  currentFile = file;
  currentClassifier = $("classifier").value;
  preview.src = URL.createObjectURL(file);
  box.hidden = false;
  out.hidden = false;
  out.innerHTML = "<p>Scanning image…</p>";
  const form = new FormData();
  form.append("file", file, file.name || "camera-capture.jpg");
  form.append("classifier", currentClassifier);
  try {
    const response = await fetch("/predict", { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
    renderResult(data);
  } catch (error) {
    showError(error.message || "Unable to classify this image.");
  }
}

$("btnGal").addEventListener("click", () => gal.click());
gal.addEventListener("change", () => {
  if (gal.files && gal.files[0]) classify(gal.files[0]);
});
drop.addEventListener("dragover", (event) => {
  event.preventDefault();
  drop.classList.add("dragover");
});
drop.addEventListener("dragleave", () => drop.classList.remove("dragover"));
drop.addEventListener("drop", (event) => {
  event.preventDefault();
  drop.classList.remove("dragover");
  const file = event.dataTransfer.files && event.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) classify(file);
  else showError("Please drop an image file.");
});
$("again").addEventListener("click", () => { if (currentFile) classify(currentFile); });
$("save").addEventListener("click", () => {
  if (!preview.src) return;
  const link = document.createElement("a");
  link.href = preview.src;
  link.download = currentFile?.name || "classified-image";
  link.click();
});

$("btnCam").addEventListener("click", async () => {
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: true });
    cameraVideo.srcObject = cameraStream;
    cameraPanel.hidden = false;
    drop.hidden = true;
    $("cameraMessage").textContent = "";
  } catch {
    $("cameraMessage").textContent = "Camera access is unavailable. Check browser permissions and use HTTPS or localhost.";
    cameraPanel.hidden = false;
  }
});
function closeCamera() {
  if (cameraStream) cameraStream.getTracks().forEach((track) => track.stop());
  cameraStream = null;
  cameraVideo.srcObject = null;
  cameraPanel.hidden = true;
  drop.hidden = false;
}
$("closeCamera").addEventListener("click", closeCamera);
$("takeSnapshot").addEventListener("click", () => {
  if (!cameraVideo.videoWidth) {
    $("cameraMessage").textContent = "Camera is not ready yet.";
    return;
  }
  cameraCanvas.width = cameraVideo.videoWidth;
  cameraCanvas.height = cameraVideo.videoHeight;
  cameraCanvas.getContext("2d").drawImage(cameraVideo, 0, 0);
  cameraCanvas.toBlob((blob) => {
    if (blob) {
      closeCamera();
      classify(new File([blob], "camera-capture.jpg", { type: "image/jpeg" }));
    }
  }, "image/jpeg", 0.92);
});

// Theme state is stored on <html> because the stylesheet uses html[data-theme].
const themeToggle = $("themeToggle");
const THEME_STORAGE_KEY = "cnn-image-analysis-theme";
function applyTheme(theme) {
  const selectedTheme = theme === "light" ? "light" : "dark";
  document.documentElement.dataset.theme = selectedTheme;
  if (themeToggle) {
    themeToggle.textContent = selectedTheme === "light" ? "Dark mode" : "Light mode";
    themeToggle.setAttribute("aria-label", `Switch to ${selectedTheme === "light" ? "dark" : "light"} mode`);
    themeToggle.setAttribute("aria-pressed", String(selectedTheme === "light"));
  }
}
let savedTheme = "dark";
try {
  savedTheme = localStorage.getItem(THEME_STORAGE_KEY) || "dark";
} catch (_) {
  // The page still works if browser storage is unavailable.
}
applyTheme(savedTheme);
if (themeToggle) {
  themeToggle.addEventListener("click", () => {
    const nextTheme = document.documentElement.dataset.theme === "light" ? "dark" : "light";
    applyTheme(nextTheme);
    try {
      localStorage.setItem(THEME_STORAGE_KEY, nextTheme);
    } catch (_) {
      // Theme remains active for this page load if browser storage is unavailable.
    }
  });
}
