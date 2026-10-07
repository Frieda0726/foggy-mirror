import { FaceLandmarker, FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import type { FaceLandmarkerResult, HandLandmarkerResult, NormalizedLandmark } from '@mediapipe/tasks-vision';
import { FogRenderer } from './FogRenderer';
import { createWorker, PSM, type Worker } from 'tesseract.js';
import './style.css';

const video = required<HTMLVideoElement>('camera');
const canvas = required<HTMLCanvasElement>('mirror');
const mask = document.createElement('canvas');
const maskCtx = mask.getContext('2d', { alpha: true })!;
const renderer = new FogRenderer(canvas, video);
const startButton = required<HTMLButtonElement>('startButton');
const resetButton = required<HTMLButtonElement>('resetButton');
const clearButton = required<HTMLButtonElement>('clearButton');
const beautifyButton = required<HTMLButtonElement>('beautifyButton');
const permission = required<HTMLElement>('permission');
const cursor = required<HTMLElement>('cursor');
const breathPulse = required<HTMLElement>('breathPulse');
const writeCalibration = required<HTMLElement>('writeCalibration');
const handState = required<HTMLElement>('handState');
const recognitionState = required<HTMLElement>('recognitionState');
const faceState = required<HTMLElement>('faceState');
const toast = required<HTMLElement>('toast');

let handLandmarker: HandLandmarker | null = null;
let faceLandmarker: FaceLandmarker | null = null;
let running = false;
let lastVideoTime = -1;
let lastFinger: Point | null = null;
let smoothFinger: Point | null = null;
let fingerArmed = false;
let lastFingerAt = 0;
let handCalibrationStartedAt = 0;
let handCalibrated = false;
let calibratedPalmWidth = 150;
let calibratedMoveSpeed = 0.32;
const palmSamples: number[] = [];
const speedSamples: number[] = [];
let calibrationPoint: Point | null = null;
let calibrationPointAt = 0;
let currentStroke: Point[] = [];
let pendingWordStrokes: Point[][] = [];
let recognitionTimer = 0;
let recognitionWorker: Worker | null = null;
let recognitionWorkerPromise: Promise<Worker> | null = null;
let recognitionBusy = false;
let beautifyEnabled = true;
let lastLiveRecognitionAt = 0;
let penRepositioning = false;
let lastLiveCandidate = '';
let liveCandidateHits = 0;
let puckerStartedAt = 0;
let lastBreathAt = 0;
let mouthBaseline = 0;
let mouthCalibrationFrames = 0;
let toastTimer = 0;
let lastRecoveryAt = 0;
const breathClouds: BreathCloud[] = [];

type Point = { x: number; y: number };
type BreathCloud = { origin: Point; radius: number; startedAt: number; emitted: number; seed: number };

function required<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`Missing #${id}`);
  return node as T;
}

function resize(): void {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  renderer.resize(innerWidth, innerHeight, dpr);
  mask.width = canvas.width;
  mask.height = canvas.height;
  maskCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function resetFog(): void {
  maskCtx.save();
  maskCtx.setTransform(1, 0, 0, 1, 0, 0);
  maskCtx.clearRect(0, 0, mask.width, mask.height);
  maskCtx.restore();
  showToast('The mirror has fogged up');
}

function clearAllFog(): void {
  maskCtx.save();
  maskCtx.fillStyle = '#fff';
  maskCtx.fillRect(0, 0, innerWidth, innerHeight);
  maskCtx.restore();
  showToast('Mirror cleared');
}

function stroke(from: Point | null, to: Point, radius = 18): void {
  maskCtx.save();
  maskCtx.strokeStyle = '#fff';
  maskCtx.fillStyle = '#fff';
  maskCtx.lineWidth = radius * 2;
  maskCtx.lineCap = 'round';
  maskCtx.lineJoin = 'round';
  if (from) {
    maskCtx.beginPath();
    maskCtx.moveTo(from.x, from.y);
    maskCtx.lineTo(to.x, to.y);
    maskCtx.stroke();
  } else {
    maskCtx.beginPath();
    maskCtx.arc(to.x, to.y, radius, 0, Math.PI * 2);
    maskCtx.fill();
  }
  maskCtx.restore();
}

function clearFogWithBreath(point: Point, radius: number, strength = 0.26): void {
  maskCtx.save();
  const gradient = maskCtx.createRadialGradient(point.x, point.y, radius * 0.12, point.x, point.y, radius);
  gradient.addColorStop(0, `rgba(255,255,255,${strength})`);
  gradient.addColorStop(0.46, `rgba(255,255,255,${strength * 0.7})`);
  gradient.addColorStop(0.78, `rgba(255,255,255,${strength * 0.22})`);
  gradient.addColorStop(1, 'rgba(255,255,255,0)');
  maskCtx.fillStyle = gradient;
  maskCtx.beginPath();
  maskCtx.arc(point.x, point.y, radius, 0, Math.PI * 2);
  maskCtx.fill();
  maskCtx.restore();
}

function seededRandom(seed: number): number {
  return Math.abs(Math.sin(seed * 91.733 + 17.17) * 43758.5453) % 1;
}

function updateBreathClouds(now: number): void {
  for (let cloudIndex = breathClouds.length - 1; cloudIndex >= 0; cloudIndex--) {
    const cloud = breathClouds[cloudIndex]!;
    const progress = Math.min(1, (now - cloud.startedAt) / 900);
    const targetCount = Math.floor(progress * 42);
    while (cloud.emitted < targetCount) {
      const index = cloud.emitted++;
      const angle = seededRandom(cloud.seed + index * 3.1) * Math.PI * 2;
      const spread = Math.sqrt(seededRandom(cloud.seed + index * 5.7));
      const horizontal = Math.cos(angle) * cloud.radius * spread * 0.68;
      const vertical = Math.sin(angle) * cloud.radius * spread * 0.4 - cloud.radius * progress * 0.08;
      const puffRadius = cloud.radius * (0.1 + seededRandom(cloud.seed + index * 7.9) * 0.14);
      const strength = 0.13 + seededRandom(cloud.seed + index * 11.3) * 0.13;
      clearFogWithBreath(
        { x: cloud.origin.x + horizontal, y: cloud.origin.y + vertical },
        puffRadius,
        strength
      );
    }
    if (progress >= 1) breathClouds.splice(cloudIndex, 1);
  }
}

function render(now = performance.now()): void {
  renderer.render(mask, now);
}

function recoverCondensation(now: number): void {
  if (now - lastRecoveryAt < 80) return;
  lastRecoveryAt = now;
  maskCtx.save();
  maskCtx.globalCompositeOperation = 'destination-out';
  maskCtx.fillStyle = 'rgba(0,0,0,0.012)';
  maskCtx.fillRect(0, 0, innerWidth, innerHeight);
  maskCtx.restore();
}

function median(values: number[]): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)]!;
}

function finishStroke(): void {
  if (currentStroke.length < 2) {
    currentStroke = [];
    return;
  }
  if (currentStroke.length >= 18 && looksLikeHeart(currentStroke)) {
    showToast('♡ Heart detected');
    pendingWordStrokes = [];
    window.clearTimeout(recognitionTimer);
  } else if (beautifyEnabled) {
    pendingWordStrokes.push(currentStroke.map((point) => ({ ...point })));
    window.clearTimeout(recognitionTimer);
    recognitionTimer = window.setTimeout(() => void beautifyPendingWord(), 550);
  }
  currentStroke = [];
}

async function getRecognitionWorker(): Promise<Worker> {
  if (recognitionWorker) return recognitionWorker;
  if (!recognitionWorkerPromise) {
    recognitionWorkerPromise = (async () => {
      const worker = await createWorker('eng');
      await worker.setParameters({
        tessedit_pageseg_mode: PSM.SINGLE_WORD,
        tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789',
        preserve_interword_spaces: '1',
        user_defined_dpi: '300',
      });
      recognitionWorker = worker;
      return worker;
    })();
  }
  return recognitionWorkerPromise;
}

async function beautifyPendingWord(): Promise<void> {
  if (!beautifyEnabled || !pendingWordStrokes.length) return;
  if (recognitionBusy) {
    recognitionTimer = window.setTimeout(() => void beautifyPendingWord(), 280);
    return;
  }
  const strokes = pendingWordStrokes;
  pendingWordStrokes = [];
  const points = strokes.flat();
  const left = Math.min(...points.map((point) => point.x));
  const right = Math.max(...points.map((point) => point.x));
  const top = Math.min(...points.map((point) => point.y));
  const bottom = Math.max(...points.map((point) => point.y));
  const width = right - left;
  const height = bottom - top;
  if (width < 18 || height < 18) return;

  recognitionBusy = true;
  try {
    const padding = Math.max(24, height * 0.35);
    const scale = Math.min(2.6, 420 / Math.max(width + padding * 2, height + padding * 2));
    const sample = document.createElement('canvas');
    sample.width = Math.ceil((width + padding * 2) * scale);
    sample.height = Math.ceil((height + padding * 2) * scale);
    const context = sample.getContext('2d')!;
    context.fillStyle = '#fff';
    context.fillRect(0, 0, sample.width, sample.height);
    context.strokeStyle = '#000';
    // OCR works substantially better with normalized, pen-like strokes than
    // with the deliberately thick cleared paths used by the fog renderer.
    context.lineWidth = Math.max(4, Math.min(10, calibratedPalmWidth * 0.055)) * scale;
    context.lineCap = 'round';
    context.lineJoin = 'round';
    for (const line of strokes) {
      context.beginPath();
      line.forEach((point, index) => {
        const x = (point.x - left + padding) * scale;
        const y = (point.y - top + padding) * scale;
        if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
      });
      context.stroke();
    }

    const worker = await getRecognitionWorker();
    const result = await worker.recognize(sample);
    const text = result.data.text.replace(/[^A-Za-z0-9 ]/g, '').trim();
    if (!text || result.data.confidence < 62) {
      setState(recognitionState, false, 'Original strokes preserved');
      return;
    }

    const areaPadding = Math.max(18, height * 0.22);
    maskCtx.save();
    maskCtx.globalCompositeOperation = 'destination-out';
    maskCtx.fillStyle = '#000';
    maskCtx.fillRect(left - areaPadding, top - areaPadding, width + areaPadding * 2, height + areaPadding * 2);
    maskCtx.restore();
    maskCtx.save();
    maskCtx.fillStyle = '#fff';
    maskCtx.textAlign = 'center';
    maskCtx.textBaseline = 'middle';
    const fontSize = Math.max(42, Math.min(150, height * 1.18));
    maskCtx.font = `500 ${fontSize}px "Bradley Hand", "Segoe Print", "KaiTi", cursive`;
    maskCtx.fillText(text, (left + right) / 2, (top + bottom) / 2, width + areaPadding);
    maskCtx.restore();
    setState(recognitionState, true, `Recognized · ${text}`);
  } catch (error) {
    console.error(error);
  } finally {
    recognitionBusy = false;
    beautifyButton.textContent = 'Smart script';
  }
}

async function analyzeWritingLive(): Promise<void> {
  if (!beautifyEnabled || recognitionBusy || currentStroke.length < 10) return;
  const strokes = [...pendingWordStrokes, currentStroke].map((line) => line.map((point) => ({ ...point })));
  const points = strokes.flat();
  const left = Math.min(...points.map((point) => point.x));
  const right = Math.max(...points.map((point) => point.x));
  const top = Math.min(...points.map((point) => point.y));
  const bottom = Math.max(...points.map((point) => point.y));
  const width = right - left;
  const height = bottom - top;
  if (width < 18 || height < 18) return;

  recognitionBusy = true;
  setState(recognitionState, true, 'Analyzing handwriting…');
  try {
    const padding = Math.max(24, height * 0.35);
    const scale = Math.min(2.5, 420 / Math.max(width + padding * 2, height + padding * 2));
    const sample = document.createElement('canvas');
    sample.width = Math.ceil((width + padding * 2) * scale);
    sample.height = Math.ceil((height + padding * 2) * scale);
    const context = sample.getContext('2d')!;
    context.fillStyle = '#fff';
    context.fillRect(0, 0, sample.width, sample.height);
    context.strokeStyle = '#000';
    context.lineWidth = Math.max(4, Math.min(10, calibratedPalmWidth * 0.055)) * scale;
    context.lineCap = 'round';
    context.lineJoin = 'round';
    for (const line of strokes) {
      context.beginPath();
      line.forEach((point, index) => {
        const x = (point.x - left + padding) * scale;
        const y = (point.y - top + padding) * scale;
        if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
      });
      context.stroke();
    }
    const worker = await getRecognitionWorker();
    const result = await worker.recognize(sample);
    const candidate = result.data.text.replace(/[^A-Za-z0-9 ]/g, '').trim();
    if (candidate && candidate.toLowerCase() === lastLiveCandidate.toLowerCase()) {
      liveCandidateHits += 1;
    } else {
      lastLiveCandidate = candidate;
      liveCandidateHits = candidate ? 1 : 0;
    }
    const stableCandidate = candidate && result.data.confidence >= 42 && liveCandidateHits >= 2;
    setState(
      recognitionState,
      Boolean(stableCandidate),
      stableCandidate ? `Possible · ${candidate}` : 'Analyzing strokes…'
    );
  } catch (error) {
    console.error(error);
    setState(recognitionState, false, 'Recognition unavailable');
  } finally {
    recognitionBusy = false;
  }
}

function looksLikeHeart(points: Point[]): boolean {
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const left = Math.min(...xs);
  const right = Math.max(...xs);
  const top = Math.min(...ys);
  const bottom = Math.max(...ys);
  const width = right - left;
  const height = bottom - top;
  if (width < 70 || height < 70 || width / height < 0.62 || width / height > 1.55) return false;
  const centerX = (left + right) / 2;
  const closure = Math.hypot(points[0]!.x - points.at(-1)!.x, points[0]!.y - points.at(-1)!.y);
  const bottomPoint = points.reduce((lowest, point) => point.y > lowest.y ? point : lowest);
  const leftLobe = points.filter((point) => point.x < centerX - width * 0.12 && point.y < top + height * 0.48);
  const rightLobe = points.filter((point) => point.x > centerX + width * 0.12 && point.y < top + height * 0.48);
  const notch = points.filter((point) => Math.abs(point.x - centerX) < width * 0.18 && point.y < top + height * 0.48);
  if (!leftLobe.length || !rightLobe.length || !notch.length) return false;
  const lobeTop = (Math.min(...leftLobe.map((point) => point.y)) + Math.min(...rightLobe.map((point) => point.y))) / 2;
  const notchDepth = Math.max(...notch.map((point) => point.y)) - lobeTop;
  return closure < Math.max(width, height) * 0.38
    && Math.abs(bottomPoint.x - centerX) < width * 0.24
    && notchDepth > height * 0.07;
}

function updateHand(result: HandLandmarkerResult): void {
  const landmarks = result.landmarks[0];
  if (!landmarks) {
    finishStroke();
    lastFinger = null;
    smoothFinger = null;
    fingerArmed = false;
    lastFingerAt = 0;
    penRepositioning = false;
    cursor.style.opacity = '0';
    setState(handState, false, 'No hand detected');
    return;
  }
  const tip = landmarks[8]!;
  const raw = { x: (1 - tip.x) * innerWidth, y: tip.y * innerHeight };
  smoothFinger = smoothFinger
    ? { x: smoothFinger.x * 0.55 + raw.x * 0.45, y: smoothFinger.y * 0.55 + raw.y * 0.45 }
    : raw;
  const now = performance.now();
  const palmSpan = Math.max(0.001, distance(landmarks[5]!, landmarks[17]!));
  const palmWidth = palmSpan * innerWidth;
  if (!handCalibrated) {
    if (!handCalibrationStartedAt) {
      handCalibrationStartedAt = now;
      writeCalibration.classList.add('show');
    }
    palmSamples.push(palmWidth);
    if (calibrationPointAt && calibrationPoint) {
      const sampleTravel = Math.hypot(smoothFinger.x - calibrationPoint.x, smoothFinger.y - calibrationPoint.y);
      const sampleElapsed = Math.max(1, now - calibrationPointAt);
      if (sampleTravel > 1) speedSamples.push(sampleTravel / sampleElapsed);
    }
    calibrationPoint = { ...smoothFinger };
    calibrationPointAt = now;
    if (now - handCalibrationStartedAt >= 1200 && palmSamples.length >= 12) {
      calibratedPalmWidth = median(palmSamples) || palmWidth;
      calibratedMoveSpeed = median(speedSamples) || 0.32;
      handCalibrated = true;
      writeCalibration.classList.add('done');
      showToast('Pinch calibrated · Ready to write');
    }
  }
  const pinchRatio = distance(landmarks[4]!, landmarks[8]!) / palmSpan;
  const pinching = fingerArmed ? pinchRatio < 0.58 : pinchRatio < 0.38;
  if (pinching && !fingerArmed) {
    fingerArmed = true;
    lastFinger = null;
  } else if (!pinching && fingerArmed) {
    fingerArmed = false;
    finishStroke();
    lastFinger = null;
  }
  const writing = fingerArmed && handCalibrated;
  cursor.style.opacity = '1';
  cursor.style.transform = `translate(${smoothFinger.x}px, ${smoothFinger.y}px)`;
  cursor.classList.toggle('drawing', writing);
  setState(handState, true, writing ? 'Writing · Pinched' : `Hovering · Pinch to write (${Math.round(pinchRatio * 100)}%)`);
  handState.classList.toggle('drawing', writing);
  if (writing) {
    const elapsed = lastFingerAt ? Math.max(1, now - lastFingerAt) : 16;
    const travel = lastFinger ? Math.hypot(smoothFinger.x - lastFinger.x, smoothFinger.y - lastFinger.y) : 0;
    const speed = travel / elapsed;
    // Preserve continuous strokes even when the fingertip moves quickly. A jump is
    // only treated as repositioning when tracking has clearly skipped a large gap.
    const isRepositioning = travel > Math.max(62, calibratedPalmWidth * 0.42) || speed > Math.max(1.85, calibratedMoveSpeed * 6);
    if (isRepositioning) {
      lastFinger = null;
      penRepositioning = true;
      lastFingerAt = now;
      return;
    }
    if (penRepositioning) {
      penRepositioning = false;
      lastFinger = null;
    }
    stroke(lastFinger, smoothFinger, Math.max(12, Math.min(22, calibratedPalmWidth * 0.13)));
    currentStroke.push({ ...smoothFinger });
      if (now - lastLiveRecognitionAt > 500 && currentStroke.length >= 10) {
      lastLiveRecognitionAt = now;
      void analyzeWritingLive();
    }
    lastFinger = smoothFinger;
    lastFingerAt = now;
  } else {
    finishStroke();
    lastFinger = null;
    lastFingerAt = 0;
    penRepositioning = false;
    lastLiveCandidate = '';
    liveCandidateHits = 0;
  }
}

function updateFace(result: FaceLandmarkerResult, now: number): void {
  const face = result.faceLandmarks[0];
  const scores = result.faceBlendshapes[0]?.categories;
  const pucker = scores?.find((item) => item.categoryName === 'mouthPucker')?.score ?? 0;
  const funnel = scores?.find((item) => item.categoryName === 'mouthFunnel')?.score ?? 0;
  if (!face) {
    puckerStartedAt = 0;
    mouthCalibrationFrames = 0;
    setState(faceState, false, 'No face detected');
    return;
  }
  const mouthWidth = Math.max(0.001, distance(face[61]!, face[291]!));
  const mouthOpening = distance(face[13]!, face[14]!);
  const mouthAspect = mouthOpening / mouthWidth;
  if (!mouthBaseline) mouthBaseline = mouthAspect;
  const looksClosed = mouthAspect < Math.max(0.105, mouthBaseline * 1.35);
  if (!puckerStartedAt && looksClosed) {
    mouthBaseline = mouthBaseline * 0.94 + mouthAspect * 0.06;
    mouthCalibrationFrames = Math.min(90, mouthCalibrationFrames + 1);
  }
  const calibrated = mouthCalibrationFrames >= 24;
  const openThreshold = Math.max(0.135, mouthBaseline * 1.72, mouthBaseline + 0.065);
  const geometryScore = Math.max(0, Math.min(1, (mouthAspect - openThreshold) / 0.12));
  const breathScore = Math.max(pucker, funnel * 1.1, geometryScore);
  const shapeSupportsBreath = pucker > 0.18 || funnel > 0.14 || mouthAspect > openThreshold + 0.035;
  const active = calibrated && mouthAspect > openThreshold && shapeSupportsBreath;
  const holdProgress = puckerStartedAt ? Math.min(100, Math.round(((now - puckerStartedAt) / 480) * 100)) : 0;
  const confidence = Math.round(breathScore * 100);
  setState(
    faceState,
    true,
    !calibrated ? `Calibrating breath · ${Math.round((mouthCalibrationFrames / 24) * 100)}%` : active ? `Detecting breath · ${holdProgress}%` : `Breath ${confidence}% · Open your mouth`
  );
  if (!active) {
    puckerStartedAt = 0;
    return;
  }
  if (!puckerStartedAt) puckerStartedAt = now;
  if (now - puckerStartedAt > 480 && now - lastBreathAt > 1400) {
    const upperLip = face[13]!;
    const lowerLip = face[14]!;
    const mouth = { x: (1 - (upperLip.x + lowerLip.x) / 2) * innerWidth, y: ((upperLip.y + lowerLip.y) / 2) * innerHeight };
    const faceCenter = face[1]
      ? { x: (1 - face[1]!.x) * innerWidth, y: face[1]!.y * innerHeight }
      : mouth;
    const faceWidth = distance(face[234]!, face[454]!) * innerWidth;
    const breathRadius = Math.max(95, Math.min(210, faceWidth * 0.62));
    breathClouds.push({
      origin: { x: (mouth.x + faceCenter.x) / 2, y: (mouth.y + faceCenter.y) / 2 },
      radius: breathRadius,
      startedAt: now,
      emitted: 0,
      seed: now * 0.013,
    });
    breathPulse.style.left = `${mouth.x}px`;
    breathPulse.style.top = `${mouth.y}px`;
    breathPulse.classList.remove('play');
    void breathPulse.offsetWidth;
    breathPulse.classList.add('play');
    lastBreathAt = now;
    puckerStartedAt = 0;
    showToast('Whoosh — the mist cleared');
  }
}

function distance(a: NormalizedLandmark, b: NormalizedLandmark): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function setState(node: HTMLElement, active: boolean, text: string): void {
  node.classList.toggle('active', active);
  node.querySelector('span')!.textContent = text;
}

function showToast(message: string): void {
  toast.textContent = message;
  toast.classList.add('show');
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => toast.classList.remove('show'), 1800);
}

async function initializeModels(): Promise<void> {
  const vision = await FilesetResolver.forVisionTasks('/wasm');
  [handLandmarker, faceLandmarker] = await Promise.all([
    HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numHands: 1,
      minHandDetectionConfidence: 0.55,
      minHandPresenceConfidence: 0.55,
      minTrackingConfidence: 0.5,
    }),
    FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numFaces: 1,
      outputFaceBlendshapes: true,
      minFaceDetectionConfidence: 0.55,
      minFacePresenceConfidence: 0.55,
      minTrackingConfidence: 0.5,
    }),
  ]);
}

async function start(): Promise<void> {
  startButton.disabled = true;
  startButton.textContent = 'Preparing on-device models…';
  try {
    const [stream] = await Promise.all([
      navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false }),
      initializeModels(),
    ]);
    video.srcObject = stream;
    await video.play();
    permission.classList.add('hidden');
    document.body.classList.add('running');
    running = true;
    if (beautifyEnabled) void getRecognitionWorker().catch(() => setState(recognitionState, false, 'Recognition model unavailable'));
    showToast('Ready · Pinch to start writing');
    requestAnimationFrame(loop);
  } catch (error) {
    console.error(error);
    startButton.disabled = false;
    startButton.textContent = 'Try camera again';
    showToast('Could not start · Check camera access and connection');
  }
}

function loop(now: number): void {
  if (!running) return;
  if (video.readyState >= 2 && video.currentTime !== lastVideoTime && handLandmarker && faceLandmarker) {
    lastVideoTime = video.currentTime;
    updateHand(handLandmarker.detectForVideo(video, now));
    updateFace(faceLandmarker.detectForVideo(video, now), now);
  }
  recoverCondensation(now);
  updateBreathClouds(now);
  render(now);
  requestAnimationFrame(loop);
}

let pointerDown = false;
canvas.addEventListener('pointerdown', (event) => { pointerDown = true; lastFinger = { x: event.clientX, y: event.clientY }; stroke(null, lastFinger, 18); });
canvas.addEventListener('pointermove', (event) => { if (!pointerDown) return; const next = { x: event.clientX, y: event.clientY }; stroke(lastFinger, next, 18); lastFinger = next; });
window.addEventListener('pointerup', () => { pointerDown = false; lastFinger = null; });
window.addEventListener('resize', () => { resize(); resetFog(); });
startButton.addEventListener('click', start);
resetButton.addEventListener('click', resetFog);
clearButton.addEventListener('click', clearAllFog);
beautifyButton.addEventListener('click', () => {
  beautifyEnabled = !beautifyEnabled;
  beautifyButton.classList.toggle('active', beautifyEnabled);
  beautifyButton.textContent = 'Smart script';
  setState(recognitionState, beautifyEnabled, beautifyEnabled ? 'Recognition ready' : 'Recognition off');
  if (!beautifyEnabled) {
    pendingWordStrokes = [];
    window.clearTimeout(recognitionTimer);
  }
});

resize();
render();
// Preload the local recognizer while the permission screen is visible so the
// first beautification can respond immediately after the user starts writing.
if (beautifyEnabled) void getRecognitionWorker().catch(() => undefined);
