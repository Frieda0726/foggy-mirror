import { FaceLandmarker, FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import type { FaceLandmarkerResult, HandLandmarkerResult, NormalizedLandmark } from '@mediapipe/tasks-vision';
import { FogRenderer } from './FogRenderer';
import './style.css';

const video = required<HTMLVideoElement>('camera');
const canvas = required<HTMLCanvasElement>('mirror');
const mask = document.createElement('canvas');
const maskCtx = mask.getContext('2d', { alpha: true })!;
const renderer = new FogRenderer(canvas, video);
const startButton = required<HTMLButtonElement>('startButton');
const resetButton = required<HTMLButtonElement>('resetButton');
const clearButton = required<HTMLButtonElement>('clearButton');
const permission = required<HTMLElement>('permission');
const cursor = required<HTMLElement>('cursor');
const breathPulse = required<HTMLElement>('breathPulse');
const writeCalibration = required<HTMLElement>('writeCalibration');
const handState = required<HTMLElement>('handState');
const faceState = required<HTMLElement>('faceState');
const toast = required<HTMLElement>('toast');

let handLandmarker: HandLandmarker | null = null;
let faceLandmarker: FaceLandmarker | null = null;
let running = false;
let lastVideoTime = -1;
let lastFinger: Point | null = null;
let smoothFinger: Point | null = null;
let fingerHoldStartedAt = 0;
let fingerArmed = false;
let lastFingerAt = 0;
let strokePausedAt = 0;
let handCalibrationStartedAt = 0;
let handCalibrated = false;
let calibratedPalmWidth = 150;
let calibratedMoveSpeed = 0.32;
const palmSamples: number[] = [];
const speedSamples: number[] = [];
let calibrationPoint: Point | null = null;
let calibrationPointAt = 0;
let currentStroke: Point[] = [];
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
  showToast('镜面已重新起雾');
}

function clearAllFog(): void {
  maskCtx.save();
  maskCtx.fillStyle = '#fff';
  maskCtx.fillRect(0, 0, innerWidth, innerHeight);
  maskCtx.restore();
  showToast('雾气已清空');
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

function isFingerWriting(landmarks: NormalizedLandmark[]): boolean {
  const indexTip = landmarks[8];
  return Boolean(indexTip && Number.isFinite(indexTip.x) && Number.isFinite(indexTip.y));
}

function median(values: number[]): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)]!;
}

function finishStroke(): void {
  if (currentStroke.length >= 18 && looksLikeHeart(currentStroke)) {
    showToast('♡ 识别到爱心');
  }
  currentStroke = [];
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
    fingerHoldStartedAt = 0;
    fingerArmed = false;
    lastFingerAt = 0;
    strokePausedAt = 0;
    cursor.style.opacity = '0';
    setState(handState, false, '未检测到手 · 伸出食指');
    return;
  }
  const tip = landmarks[8]!;
  const raw = { x: (1 - tip.x) * innerWidth, y: tip.y * innerHeight };
  smoothFinger = smoothFinger
    ? { x: smoothFinger.x * 0.55 + raw.x * 0.45, y: smoothFinger.y * 0.55 + raw.y * 0.45 }
    : raw;
  const pointing = isFingerWriting(landmarks);
  const now = performance.now();
  const palmWidth = distance(landmarks[5]!, landmarks[17]!) * innerWidth;
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
      showToast('书写已校准 · 可以写字或画爱心');
    }
  }
  if (pointing && !fingerArmed) {
    if (!fingerHoldStartedAt) {
      fingerHoldStartedAt = now;
    } else if (now - fingerHoldStartedAt > 350) {
      fingerArmed = true;
      lastFinger = null;
      showToast('已落笔 · 移动食指书写');
    }
  } else if (!pointing) {
    fingerHoldStartedAt = 0;
    fingerArmed = false;
  }
  const writing = pointing && fingerArmed && handCalibrated;
  cursor.style.opacity = '1';
  cursor.style.transform = `translate(${smoothFinger.x}px, ${smoothFinger.y}px)`;
  cursor.classList.toggle('drawing', writing);
  const readyProgress = fingerHoldStartedAt ? Math.min(100, Math.round(((now - fingerHoldStartedAt) / 350) * 100)) : 0;
  setState(handState, true, writing ? '正在书写 · 移动食指' : `识别食指 · ${readyProgress}%`);
  handState.classList.toggle('drawing', writing);
  if (writing) {
    const elapsed = lastFingerAt ? Math.max(1, now - lastFingerAt) : 16;
    const travel = lastFinger ? Math.hypot(smoothFinger.x - lastFinger.x, smoothFinger.y - lastFinger.y) : 0;
    const speed = travel / elapsed;
    // Preserve continuous strokes even when the fingertip moves quickly. A jump is
    // only treated as repositioning when tracking has clearly skipped a large gap.
    const isRepositioning = travel > Math.max(90, calibratedPalmWidth * 0.6) || speed > Math.max(3.2, calibratedMoveSpeed * 8);
    const isPaused = lastFinger && speed < 0.025;

    if (isPaused) {
      if (!strokePausedAt) strokePausedAt = now;
      if (now - strokePausedAt > 230) {
        lastFinger = null;
        finishStroke();
      }
    } else {
      if (isRepositioning) {
        lastFinger = null;
        finishStroke();
        strokePausedAt = 0;
        lastFingerAt = now;
        return;
      }
      if (strokePausedAt && now - strokePausedAt > 180) {
        lastFinger = null;
        finishStroke();
      }
      strokePausedAt = 0;
      stroke(lastFinger, smoothFinger, Math.max(12, Math.min(22, calibratedPalmWidth * 0.13)));
      currentStroke.push({ ...smoothFinger });
      lastFinger = smoothFinger;
    }
    lastFingerAt = now;
  } else {
    finishStroke();
    lastFinger = null;
    lastFingerAt = 0;
    strokePausedAt = 0;
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
    setState(faceState, false, '未检测到面部 · 正对镜头');
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
    !calibrated ? `正在校准嘴形 · ${Math.round((mouthCalibrationFrames / 24) * 100)}%` : active ? `正在识别哈气 · ${holdProgress}%` : `哈气识别 ${confidence}% · 张嘴哈气`
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
    showToast('呼——雾气被吹开了');
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
  startButton.textContent = '正在准备本地模型…';
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
    showToast('准备好了 · 伸出食指开始写字');
    requestAnimationFrame(loop);
  } catch (error) {
    console.error(error);
    startButton.disabled = false;
    startButton.textContent = '重试开启摄像头';
    showToast('无法启动，请检查摄像头权限与网络');
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

resize();
render();
