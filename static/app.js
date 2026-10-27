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
let cameraFacingMode = "environment";
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
  if (!file) return;
  currentFile = file;

  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  preview.src = previewUrl;

  box.hidden = false;
  out.hidden = false;
  out.innerHTML = "<p>Scanning image… please wait.</p>";

  const stage = $("stage");
  const againButton = $("again");
  const saveButton = $("save");
  const previousAgainText = againButton?.textContent;
  if (stage) stage.classList.add("scanning");
  if (againButton) {
    againButton.disabled = true;
    againButton.textContent = "Processing…";
  }
  if (saveButton) saveButton.disabled = true;

  const form = new FormData();
  form.append("file", file, file.name || "camera-capture.jpg");

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: form,
      cache: "no-store"
    });

    let data;
    try {
      data = await response.json();
    } catch {
      throw new Error(`The server returned an unreadable response (${response.status}).`);
    }

    if (!response.ok) {
      throw new Error(data.detail || `Request failed (${response.status})`);
    }

    renderResult(data);
  } catch (error) {
    showError(error.message || "Unable to analyse this image.");
  } finally {
    if (stage) stage.classList.remove("scanning");
    if (againButton) {
      againButton.disabled = false;
      againButton.textContent = previousAgainText || "Reprocess image";
    }
    if (saveButton) saveButton.disabled = false;
  }
}

$("btnGal").addEventListener("click", () => gal.click());

gal.addEventListener("change", () => {
  if (gal.files?.[0]) classify(gal.files[0]);
});

drop.addEventListener("dragover", (event) => {
  event.preventDefault();
  drop.classList.add("over");
});

drop.addEventListener("dragleave", () => {
  drop.classList.remove("over");
});

drop.addEventListener("drop", (event) => {
  event.preventDefault();
  drop.classList.remove("over");

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

async function startCamera(facingMode = cameraFacingMode) {
  // Stop the previous stream before requesting the other camera.
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
  }

  $("cameraMessage").textContent = "Opening camera…";
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { exact: facingMode } },
      audio: false
    });
    cameraFacingMode = facingMode;
    cameraVideo.srcObject = cameraStream;
    await cameraVideo.play();
    cameraPanel.hidden = false;
    drop.hidden = true;
    $("cameraMessage").textContent =
      facingMode === "environment"
        ? "Rear camera selected."
        : "Front camera selected.";
    const switchButton = $("switchCamera");
    if (switchButton) {
      switchButton.textContent =
        facingMode === "environment" ? "Use front camera" : "Use rear camera";
    }
  } catch (error) {
    cameraPanel.hidden = false;
    $("cameraMessage").textContent =
      "Could not switch/open that camera. Your browser or device may not expose it. " +
      "Check camera permissions and use HTTPS or localhost.";
  }
}

$("btnCam").addEventListener("click", () => {
  cameraFacingMode = "environment";
  startCamera(cameraFacingMode);
});

$("switchCamera")?.addEventListener("click", () => {
  const nextMode = cameraFacingMode === "environment" ? "user" : "environment";
  startCamera(nextMode);
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

// Both pages use the same persistent theme setting and CSS selector.
const themeToggle = $("themeToggle");
const THEME_KEY = "vision-theme";

function applyTheme(theme) {
  const selected = theme === "light" ? "light" : "dark";
  document.documentElement.dataset.theme = selected;
  if (themeToggle) {
    themeToggle.textContent = selected === "dark" ? "Light mode" : "Dark mode";
    themeToggle.setAttribute("aria-pressed", String(selected === "light"));
  }
}

let savedTheme = "dark";
try {
  savedTheme = localStorage.getItem(THEME_KEY) || "dark";
} catch {
  // The page remains usable if browser storage is unavailable.
}
applyTheme(savedTheme);

if (themeToggle) {
  themeToggle.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    applyTheme(next);
    try {
      localStorage.setItem(THEME_KEY, next);
    } catch {
      // Theme changes still work for the current page without storage.
    }
  });
}
