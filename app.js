import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const imageInput = document.querySelector("#imageInput");
const startCameraButton = document.querySelector("#startCameraButton");
const captureButton = document.querySelector("#captureButton");
const stopCameraButton = document.querySelector("#stopCameraButton");
const mirrorToggle = document.querySelector("#mirrorToggle");
const cameraFeed = document.querySelector("#cameraFeed");
const previewCanvas = document.querySelector("#previewCanvas");
const statusPanel = document.querySelector("#statusPanel");
const overallScore = document.querySelector("#overallScore");
const metricGrid = document.querySelector("#metricGrid");
const feedbackList = document.querySelector("#feedbackList");
const strengthList = document.querySelector("#strengthList");

const ctx = previewCanvas.getContext("2d");

let faceLandmarker;
let cameraStream;
let isMirrorEnabled = mirrorToggle?.checked ?? true;

const metricLabels = ["Lighting", "Sharpness", "Framing", "Pose", "Expression"];

init();

async function init() {
  try {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );

    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: MODEL_URL,
      },
      runningMode: "IMAGE",
      numFaces: 1,
      outputFaceBlendshapes: true,
    });

    setStatus("Model ready. Upload a selfie or start your camera.");
  } catch (error) {
    console.error(error);
    setStatus(
      "Could not load the face analysis model. Check your connection and reload the page."
    );
  }
}

imageInput.addEventListener("change", async (event) => {
  const [file] = event.target.files ?? [];
  if (!file) {
    return;
  }

  const image = new Image();
  image.onload = async () => {
    stopCamera();
    await analyzeSource(image, file.name);
  };
  image.src = URL.createObjectURL(file);
});

startCameraButton.addEventListener("click", startCamera);
captureButton.addEventListener("click", captureFrame);
stopCameraButton.addEventListener("click", stopCamera);
mirrorToggle?.addEventListener("change", handleMirrorToggle);

applyMirrorState();
showCanvasPreview();

async function startCamera() {
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 960 },
      },
      audio: false,
    });

    cameraFeed.srcObject = cameraStream;
    showLivePreview();
    applyMirrorState();
    captureButton.disabled = false;
    stopCameraButton.disabled = false;
    setStatus("Camera ready. Capture a frame when you like the shot.");
  } catch (error) {
    console.error(error);
    setStatus("Camera access failed. You can still upload an image.");
  }
}

function stopCamera() {
  if (cameraStream) {
    for (const track of cameraStream.getTracks()) {
      track.stop();
    }
    cameraStream = null;
  }

  cameraFeed.srcObject = null;
  cameraFeed.hidden = true;
  showCanvasPreview();
  applyMirrorState();
  captureButton.disabled = true;
  stopCameraButton.disabled = true;
}

async function captureFrame() {
  if (!cameraFeed.videoWidth || !cameraFeed.videoHeight) {
    return;
  }

  await analyzeSource(cameraFeed, "camera capture");
}

async function analyzeSource(source, label) {
  if (!faceLandmarker) {
    setStatus("Model still loading. Try again in a moment.");
    return;
  }

  const width = source.videoWidth || source.naturalWidth || source.width;
  const height = source.videoHeight || source.naturalHeight || source.height;
  resizeCanvas(width, height);
  showCanvasPreview();
  drawSourceToCanvas(source, width, height);

  setStatus(`Analyzing ${label}…`);

  const detection = faceLandmarker.detect(previewCanvas);
  const landmarks = detection.faceLandmarks?.[0];

  if (!landmarks) {
    renderDefaultCanvasMessage();
    setStatus("No clear face found. Try a better-lit front-facing selfie.");
    writeResults(null);
    return;
  }

  const metrics = scoreSelfie(landmarks, previewCanvas);
  writeResults(metrics);
  setStatus("Analysis complete. Review the feedback on the right.");
}

function resizeCanvas(width, height) {
  previewCanvas.width = width;
  previewCanvas.height = height;
}

function showLivePreview() {
  cameraFeed.hidden = false;
  previewCanvas.hidden = true;
}

function showCanvasPreview() {
  previewCanvas.hidden = false;
  if (!cameraStream) {
    cameraFeed.hidden = true;
  }
}

function handleMirrorToggle(event) {
  isMirrorEnabled = event.target.checked;
  applyMirrorState();
}

function applyMirrorState() {
  cameraFeed.classList.toggle("mirrored", isMirrorEnabled);
  previewCanvas.classList.remove("mirrored");
}

function drawSourceToCanvas(source, width, height) {
  const shouldMirror = source === cameraFeed && isMirrorEnabled;

  ctx.save();
  ctx.clearRect(0, 0, width, height);

  if (shouldMirror) {
    ctx.translate(width, 0);
    ctx.scale(-1, 1);
  }

  ctx.drawImage(source, 0, 0, width, height);
  ctx.restore();
}

function scoreSelfie(landmarks, canvas) {
  const faceBox = getFaceBox(landmarks, canvas.width, canvas.height);
  const faceWidth = faceBox.maxX - faceBox.minX;
  const faceHeight = faceBox.maxY - faceBox.minY;
  const faceCenterX = (faceBox.minX + faceBox.maxX) / 2;
  const faceCenterY = (faceBox.minY + faceBox.maxY) / 2;

  const avgBrightness = getRegionBrightness(faceBox);
  const sharpness = getRegionSharpness(faceBox);

  const leftEyeOuter = px(landmarks[33], canvas);
  const rightEyeOuter = px(landmarks[263], canvas);
  const leftEyeInner = px(landmarks[133], canvas);
  const rightEyeInner = px(landmarks[362], canvas);
  const noseTip = px(landmarks[1], canvas);
  const chin = px(landmarks[152], canvas);
  const forehead = px(landmarks[10], canvas);
  const mouthLeft = px(landmarks[61], canvas);
  const mouthRight = px(landmarks[291], canvas);
  const upperLip = px(landmarks[13], canvas);
  const lowerLip = px(landmarks[14], canvas);
  const leftUpperLid = px(landmarks[159], canvas);
  const leftLowerLid = px(landmarks[145], canvas);
  const rightUpperLid = px(landmarks[386], canvas);
  const rightLowerLid = px(landmarks[374], canvas);

  const eyeTiltDegrees =
    Math.atan2(rightEyeOuter.y - leftEyeOuter.y, rightEyeOuter.x - leftEyeOuter.x) *
    (180 / Math.PI);

  const noseOffset = Math.abs(
    noseTip.x - (leftEyeInner.x + rightEyeInner.x) / 2
  ) / faceWidth;

  const faceScale = faceHeight / canvas.height;
  const centerOffset =
    Math.hypot(faceCenterX - canvas.width / 2, faceCenterY - canvas.height / 2) /
    Math.max(canvas.width, canvas.height);
  const headroom = forehead.y / canvas.height;
  const chinRoom = (canvas.height - chin.y) / canvas.height;

  const mouthOpenness = distance(upperLip, lowerLip) / faceWidth;
  const smileWidth = distance(mouthLeft, mouthRight) / faceWidth;
  const leftEyeOpen = distance(leftUpperLid, leftLowerLid) / faceWidth;
  const rightEyeOpen = distance(rightUpperLid, rightLowerLid) / faceWidth;
  const eyeBalance = 1 - Math.min(Math.abs(leftEyeOpen - rightEyeOpen) / 0.018, 1);

  const lightingScore = clampScore(
    100 -
      Math.abs(avgBrightness - 156) * 0.9 -
      (avgBrightness < 85 ? 16 : 0) -
      (avgBrightness > 220 ? 12 : 0)
  );

  const sharpnessScore = clampScore((sharpness / 22) * 100);
  const framingScore = clampScore(
    100 -
      Math.abs(faceScale - 0.56) * 240 -
      centerOffset * 180 -
      Math.abs(headroom - 0.14) * 180 -
      Math.abs(chinRoom - 0.16) * 110
  );
  const poseScore = clampScore(100 - Math.abs(eyeTiltDegrees) * 6 - noseOffset * 230);
  const expressionScore = clampScore(
    100 -
      Math.abs(mouthOpenness - 0.022) * 2000 -
      (smileWidth < 0.16 ? 12 : 0) +
      eyeBalance * 10
  );

  const metrics = [
    {
      name: "Lighting",
      score: lightingScore,
      details:
        avgBrightness < 110
          ? "The face is a bit dim. Turn toward a window or softer front light."
          : avgBrightness > 205
            ? "Highlights are getting hot. Back off harsh direct light."
            : "Exposure looks balanced across the face.",
      positive:
        avgBrightness >= 120 && avgBrightness <= 190
          ? "The lighting sits in a flattering, readable range."
          : null,
    },
    {
      name: "Sharpness",
      score: sharpnessScore,
      details:
        sharpness < 10
          ? "The image looks soft. Clean the lens and steady the phone."
          : sharpness < 18
            ? "Sharpness is decent, but you can improve it with steadier framing."
            : "Facial detail is crisp enough for a strong selfie.",
      positive:
        sharpness >= 18 ? "The photo is clear and detailed." : null,
    },
    {
      name: "Framing",
      score: framingScore,
      details:
        faceScale < 0.42
          ? "Move closer so your face fills more of the frame."
          : faceScale > 0.7
            ? "You are cropped a little tight. Give the top and chin more room."
            : centerOffset > 0.12
              ? "Shift the camera so your face sits closer to center."
              : "The framing is balanced and easy to read.",
      positive:
        faceScale >= 0.46 && faceScale <= 0.66 && centerOffset < 0.09
          ? "Your face sits at a strong size and position in the frame."
          : null,
    },
    {
      name: "Pose",
      score: poseScore,
      details:
        Math.abs(eyeTiltDegrees) > 5
          ? "The phone is tilted. Level the horizon for a cleaner portrait."
          : noseOffset > 0.06
            ? "Your head is slightly turned. A more direct angle will read more clearly."
            : "Your face is mostly straight-on, which helps consistency.",
      positive:
        Math.abs(eyeTiltDegrees) <= 4 && noseOffset <= 0.05
          ? "The head angle looks steady and direct."
          : null,
    },
    {
      name: "Expression",
      score: expressionScore,
      details:
        mouthOpenness > 0.04
          ? "Your mouth is fairly open. A softer expression may feel more natural."
          : mouthOpenness < 0.012
            ? "The expression reads a bit tense. Relax your lips and eyes."
            : "The expression feels relaxed and camera-friendly.",
      positive:
        mouthOpenness >= 0.014 && mouthOpenness <= 0.032 && eyeBalance > 0.72
          ? "Your expression feels calm and approachable."
          : null,
    },
  ];

  const overall = Math.round(
    metrics.reduce((sum, metric) => sum + metric.score, 0) / metrics.length
  );

  return {
    overall,
    metrics,
    faceBox,
  };
}

function writeResults(result) {
  if (!result) {
    overallScore.textContent = "--";
    metricGrid.innerHTML = metricLabels
      .map(
        (label) => `<article class="metric"><span>${label}</span><strong>--</strong></article>`
      )
      .join("");
    feedbackList.innerHTML = "<li>No face metrics available yet.</li>";
    strengthList.innerHTML = "<li>Upload a clear selfie to see strengths.</li>";
    return;
  }

  overallScore.textContent = `${result.overall}`;
  metricGrid.innerHTML = result.metrics
    .map(
      (metric) =>
        `<article class="metric"><span>${metric.name}</span><strong>${metric.score}</strong></article>`
    )
    .join("");

  const feedbackItems = result.metrics
    .filter((metric) => metric.score < 82)
    .map((metric) => `<li><strong>${metric.name}:</strong> ${metric.details}</li>`);
  feedbackList.innerHTML =
    feedbackItems.join("") || "<li>No major issues. This is a solid selfie.</li>";

  const strengthItems = result.metrics
    .filter((metric) => metric.positive)
    .map((metric) => `<li><strong>${metric.name}:</strong> ${metric.positive}</li>`);
  strengthList.innerHTML =
    strengthItems.join("") || "<li>The app will add positives when a metric stands out.</li>";
}

function getFaceBox(landmarks, width, height) {
  const xs = landmarks.map((point) => point.x * width);
  const ys = landmarks.map((point) => point.y * height);
  const minX = Math.max(Math.min(...xs) - width * 0.04, 0);
  const maxX = Math.min(Math.max(...xs) + width * 0.04, width);
  const minY = Math.max(Math.min(...ys) - height * 0.06, 0);
  const maxY = Math.min(Math.max(...ys) + height * 0.06, height);

  return { minX, maxX, minY, maxY };
}

function getRegionBrightness(box) {
  const { x, y, width, height } = toBoxRect(box);
  const data = ctx.getImageData(x, y, width, height).data;
  let total = 0;

  for (let i = 0; i < data.length; i += 4) {
    total += data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114;
  }

  return total / (data.length / 4);
}

function getRegionSharpness(box) {
  const { x, y, width, height } = toBoxRect(box);
  const data = ctx.getImageData(x, y, width, height).data;
  const gray = new Float32Array(width * height);

  for (let i = 0, j = 0; i < data.length; i += 4, j += 1) {
    gray[j] = data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114;
  }

  let sum = 0;
  let sumSquares = 0;
  let count = 0;

  for (let row = 1; row < height - 1; row += 1) {
    for (let col = 1; col < width - 1; col += 1) {
      const idx = row * width + col;
      const laplacian =
        gray[idx - width] +
        gray[idx - 1] +
        gray[idx + 1] +
        gray[idx + width] -
        4 * gray[idx];
      sum += laplacian;
      sumSquares += laplacian * laplacian;
      count += 1;
    }
  }

  const mean = sum / count;
  return Math.sqrt(Math.max(sumSquares / count - mean * mean, 0));
}

function toBoxRect(box) {
  return {
    x: Math.floor(box.minX),
    y: Math.floor(box.minY),
    width: Math.max(8, Math.floor(box.maxX - box.minX)),
    height: Math.max(8, Math.floor(box.maxY - box.minY)),
  };
}

function renderDefaultCanvasMessage() {
  ctx.save();
  ctx.fillStyle = "rgba(31, 26, 23, 0.72)";
  ctx.font = `${Math.max(20, previewCanvas.width / 36)}px Space Grotesk`;
  ctx.fillText("No clear face detected", 30, 54);
  ctx.restore();
}

function setStatus(message) {
  statusPanel.textContent = message;
}

function px(point, canvas) {
  return {
    x: point.x * canvas.width,
    y: point.y * canvas.height,
  };
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function clampScore(value) {
  return Math.round(Math.max(0, Math.min(100, value)));
}
