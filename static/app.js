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
let previewUrl = null;

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  }[ch]));
}

function formatPercent(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function showError(message) {
  out.hidden = false;
  out.innerHTML = `<p class="error">${escapeHtml(message)}</p>`;
}

function renderResult(data) {
  out.hidden = false;

  const result = data.results?.[0];

  if (!result) {
    out.innerHTML = `
      <div class="classification-record">
        <div class="eyebrow">[ CLASSIFICATION RECORD ]</div>
        <h2>UNABLE TO IDENTIFY</h2>
        <p>${escapeHtml(data.message || "No suitable classification was returned.")}</p>
        <p class="room101">STATUS: SENT TO ROOM 101</p>
      </div>`;
    return;
  }

  const threshold = Number(result.threshold ?? 0.60);
  const confidence = Number(result.confidence);
  const rejected = confidence < threshold ||
      result.prediction === "UNABLE TO IDENTIFY";

  const title = rejected ? "UNABLE TO IDENTIFY" : result.prediction;
  const probabilities = Object.entries(result.probabilities || {})
    .sort((a, b) => b[1] - a[1]);

  const bars = probabilities.map(([label, score]) => `
    <div class="probability-row">
      <span class="probability-label">${escapeHtml(label)}</span>
      <div class="probability-track">
        <div class="probability-fill"
             style="width:${Math.max(0, Math.min(100, Number(score) * 100))}%">
        </div>
      </div>
      <span class="probability-value">${formatPercent(score)}</span>
    </div>
  `).join("");

  out.innerHTML = `
    <div class="classification-record">
      <div class="eyebrow">
        [ CLASSIFICATION RECORD ] ${escapeHtml(result.model || "IMAGE")}
      </div>
      <h2>${escapeHtml(title)}</h2>
      <p class="confidence">${formatPercent(confidence)} confidence</p>

      ${
        rejected
          ? `<p class="room101">STATUS: ${escapeHtml(
         result.status || "SENT TO ROOM 101"
       )}</p>
       <p>${escapeHtml(
         result.message ||
         "No prediction reached the required confidence threshold."
       )}</p>`
    : `<p>${escapeHtml(result.status || "Prediction accepted.")}</p>`
      }

      <p class="threshold">
        Required confidence: ${formatPercent(threshold)}
      </p>

      <div class="probabilities">${bars}</div>
    </div>`;
}

async function classify(file) {
  currentFile = file;

  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  preview.src = previewUrl;

  box.hidden = false;
  out.hidden = false;
  out.innerHTML = "<p>Scanning image…</p>";

  const form = new FormData();
  form.append("file", file, file.name || "camera-capture.jpg");

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: form
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || `Request failed (${response.status})`);
    }

    renderResult(data);
  } catch (error) {
    showError(error.message || "Unable to analyse this image.");
  }
}

$("btnGal").addEventListener("click", () => gal.click());

gal.addEventListener("change", () => {
  if (gal.files?.[0]) classify(gal.files[0]);
});

drop.addEventListener("dragover", (event) => {
  event.preventDefault();
  drop.classList.add("dragover");
});

drop.addEventListener("dragleave", () => {
  drop.classList.remove("dragover");
});

drop.addEventListener("drop", (event) => {
  event.preventDefault();
  drop.classList.remove("dragover");

  const file = event.dataTransfer.files?.[0];

  if (file && file.type.startsWith("image/")) {
    classify(file);
  } else {
    showError("Please drop an image file.");
  }
});

$("again").addEventListener("click", () => {
  if (currentFile) classify(currentFile);
});

$("save").addEventListener("click", () => {
  if (!previewUrl) return;

  const link = document.createElement("a");
  link.href = previewUrl;
  link.download = currentFile?.name || "classified-image";
  link.click();
});

$("btnCam").addEventListener("click", async () => {
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: true
    });

    cameraVideo.srcObject = cameraStream;
    cameraPanel.hidden = false;
    drop.hidden = true;
    $("cameraMessage").textContent = "";
  } catch {
    cameraPanel.hidden = false;
    $("cameraMessage").textContent =
      "Camera unavailable. Check browser permissions and use HTTPS or localhost.";
  }
});

function closeCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
  }

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

  cameraCanvas.getContext("2d").drawImage(
    cameraVideo, 0, 0
  );

  cameraCanvas.toBlob((blob) => {
    if (!blob) return;

    closeCamera();

    classify(new File(
      [blob],
      "camera-capture.jpg",
      { type: "image/jpeg" }
    ));
  }, "image/jpeg", 0.92);
});

const themeToggle = $("themeToggle");

if (themeToggle) {
  themeToggle.addEventListener("click", () => {
    document.body.classList.toggle("light-mode");

    themeToggle.textContent =
      document.body.classList.contains("light-mode")
        ? "Dark mode"
        : "Light mode";
  });
}
