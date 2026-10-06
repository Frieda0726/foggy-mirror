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
const handState = required<HTMLElement>('handState');
const faceState = required<HTMLElement>('faceState');
const toast = required<HTMLElement>('toast');

let handLandmarker: HandLandmarker | null = null;
let faceLandmarker: FaceLandmarker | null = null;
let running = false;
let lastVideoTime = -1;
let lastFinger: Point | null = null;
let smoothFinger: Point | null = null;
let fingerAnchor: Point | null = null;
let fingerHoldStartedAt = 0;
let fingerArmed = false;
let puckerStartedAt = 0;
let lastBreathAt = 0;
let mouthBaseline = 0.065;
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
      const horizontal = Math.cos(angle) * cloud.radius * spread * 0.92;
      const vertical = Math.sin(angle) * cloud.radius * spread * 0.52 - cloud.radius * progress * 0.12;
      const puffRadius = cloud.radius * (0.14 + seededRandom(cloud.seed + index * 7.9) * 0.22);
      const strength = 0.16 + seededRandom(cloud.seed + index * 11.3) * 0.16;
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
  const indexTip = landmarks[8]!;
  const indexPip = landmarks[6]!;
  const otherTips = [landmarks[12]!, landmarks[16]!, landmarks[20]!];
  const indexExtended = indexTip.y < indexPip.y - 0.01;
  const indexIsLeading = otherTips.filter((tip) => indexTip.y + 0.012 < tip.y).length >= 2;
  return indexExtended && indexIsLeading;
}

function updateHand(result: HandLandmarkerResult): void {
  const landmarks = result.landmarks[0];
  if (!landmarks) {
    lastFinger = null;
    smoothFinger = null;
    fingerAnchor = null;
    fingerHoldStartedAt = 0;
    fingerArmed = false;
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
  if (pointing && !fingerArmed) {
    if (!fingerAnchor || distance2D(fingerAnchor, smoothFinger) > 18) {
      fingerAnchor = { ...smoothFinger };
      fingerHoldStartedAt = now;
    } else if (now - fingerHoldStartedAt > 400) {
      fingerArmed = true;
      lastFinger = null;
      showToast('已落笔 · 移动食指书写');
    }
  } else if (!pointing) {
    fingerAnchor = null;
    fingerHoldStartedAt = 0;
    fingerArmed = false;
  }
  const writing = pointing && fingerArmed;
  cursor.style.opacity = '1';
  cursor.style.transform = `translate(${smoothFinger.x}px, ${smoothFinger.y}px)`;
  cursor.classList.toggle('drawing', writing);
  const readyProgress = fingerHoldStartedAt ? Math.min(100, Math.round(((now - fingerHoldStartedAt) / 400) * 100)) : 0;
  setState(handState, true, writing ? '已落笔 · 移动食指书写' : pointing ? `保持指尖稳定 · ${readyProgress}%` : '光标模式 · 伸出食指');
  handState.classList.toggle('drawing', writing);
  if (writing) {
    const palmWidth = distance(landmarks[5]!, landmarks[17]!) * innerWidth;
    stroke(lastFinger, smoothFinger, Math.max(6, Math.min(12, palmWidth * 0.075)));
    lastFinger = smoothFinger;
  } else {
    lastFinger = null;
  }
}

function updateFace(result: FaceLandmarkerResult, now: number): void {
  const face = result.faceLandmarks[0];
  const scores = result.faceBlendshapes[0]?.categories;
  const pucker = scores?.find((item) => item.categoryName === 'mouthPucker')?.score ?? 0;
  const funnel = scores?.find((item) => item.categoryName === 'mouthFunnel')?.score ?? 0;
  if (!face) {
    puckerStartedAt = 0;
    setState(faceState, false, '未检测到面部 · 正对镜头');
    return;
  }
  const mouthWidth = Math.max(0.001, distance(face[61]!, face[291]!));
  const mouthOpening = distance(face[13]!, face[14]!);
  const mouthAspect = mouthOpening / mouthWidth;
  if (!puckerStartedAt && mouthAspect < mouthBaseline * 1.22) {
    mouthBaseline = mouthBaseline * 0.97 + mouthAspect * 0.03;
  }
  const openThreshold = Math.max(0.082, mouthBaseline * 1.28);
  const geometryScore = Math.max(0, Math.min(1, (mouthAspect - mouthBaseline) / Math.max(0.06, mouthBaseline * 1.4)));
  const breathScore = Math.max(pucker, funnel * 1.1, geometryScore);
  const active = mouthAspect > openThreshold || pucker > 0.22 || funnel > 0.16;
  const holdProgress = puckerStartedAt ? Math.min(100, Math.round(((now - puckerStartedAt) / 220) * 100)) : 0;
  const confidence = Math.round(breathScore * 100);
  setState(
    faceState,
    true,
    active ? `正在识别哈气 · ${holdProgress}%` : `哈气识别 ${confidence}% · 张嘴哈气`
  );
  if (!active) {
    puckerStartedAt = 0;
    return;
  }
  if (!puckerStartedAt) puckerStartedAt = now;
  if (now - puckerStartedAt > 220 && now - lastBreathAt > 1200) {
    const upperLip = face[13]!;
    const lowerLip = face[14]!;
    const mouth = { x: (1 - (upperLip.x + lowerLip.x) / 2) * innerWidth, y: ((upperLip.y + lowerLip.y) / 2) * innerHeight };
    const faceWidth = distance(face[234]!, face[454]!) * innerWidth;
    const breathRadius = Math.max(80, Math.min(175, faceWidth * 0.46));
    breathClouds.push({
      origin: mouth,
      radius: breathRadius * 1.2,
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

function distance2D(a: Point, b: Point): number {
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
canvas.addEventListener('pointerdown', (event) => { pointerDown = true; lastFinger = { x: event.clientX, y: event.clientY }; stroke(null, lastFinger, 9); });
canvas.addEventListener('pointermove', (event) => { if (!pointerDown) return; const next = { x: event.clientX, y: event.clientY }; stroke(lastFinger, next, 9); lastFinger = next; });
window.addEventListener('pointerup', () => { pointerDown = false; lastFinger = null; });
window.addEventListener('resize', () => { resize(); resetFog(); });
startButton.addEventListener('click', start);
resetButton.addEventListener('click', resetFog);
clearButton.addEventListener('click', clearAllFog);

resize();
render();
