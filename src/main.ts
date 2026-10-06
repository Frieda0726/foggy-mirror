import { FaceLandmarker, FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import type { FaceLandmarkerResult, HandLandmarkerResult, NormalizedLandmark } from '@mediapipe/tasks-vision';
import './style.css';

const video = required<HTMLVideoElement>('camera');
const canvas = required<HTMLCanvasElement>('mirror');
const ctx = canvas.getContext('2d', { alpha: true })!;
const mask = document.createElement('canvas');
const maskCtx = mask.getContext('2d', { alpha: true })!;
const blur = document.createElement('canvas');
const blurCtx = blur.getContext('2d', { alpha: false })!;
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
let puckerStartedAt = 0;
let lastBreathAt = 0;
let toastTimer = 0;
let noisePattern: CanvasPattern | null = null;

type Point = { x: number; y: number };

function required<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`Missing #${id}`);
  return node as T;
}

function resize(): void {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(innerWidth * dpr);
  canvas.height = Math.round(innerHeight * dpr);
  mask.width = canvas.width;
  mask.height = canvas.height;
  blur.width = Math.max(80, Math.round(innerWidth * 0.1));
  blur.height = Math.max(50, Math.round(innerHeight * 0.1));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  maskCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  createNoise();
}

function createNoise(): void {
  const tile = document.createElement('canvas');
  tile.width = tile.height = 160;
  const tileCtx = tile.getContext('2d')!;
  const image = tileCtx.createImageData(tile.width, tile.height);
  for (let i = 0; i < image.data.length; i += 4) {
    const value = 180 + Math.random() * 70;
    image.data[i] = value;
    image.data[i + 1] = value + 3;
    image.data[i + 2] = value + 5;
    image.data[i + 3] = 22 + Math.random() * 24;
  }
  tileCtx.putImageData(image, 0, 0);
  noisePattern = ctx.createPattern(tile, 'repeat');
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

function addFog(point: Point, radius: number): void {
  maskCtx.save();
  maskCtx.globalCompositeOperation = 'destination-out';
  const gradient = maskCtx.createRadialGradient(point.x, point.y, radius * 0.12, point.x, point.y, radius);
  gradient.addColorStop(0, 'rgba(0,0,0,.98)');
  gradient.addColorStop(0.48, 'rgba(0,0,0,.72)');
  gradient.addColorStop(1, 'rgba(0,0,0,0)');
  maskCtx.fillStyle = gradient;
  maskCtx.beginPath();
  maskCtx.arc(point.x, point.y, radius, 0, Math.PI * 2);
  maskCtx.fill();
  maskCtx.restore();
}

function render(): void {
  ctx.clearRect(0, 0, innerWidth, innerHeight);
  if (video.readyState >= 2) {
    blurCtx.save();
    blurCtx.translate(blur.width, 0);
    blurCtx.scale(-1, 1);
    blurCtx.drawImage(video, 0, 0, blur.width, blur.height);
    blurCtx.restore();
    ctx.drawImage(blur, 0, 0, innerWidth, innerHeight);
  }
  const fog = ctx.createLinearGradient(0, 0, 0, innerHeight);
  fog.addColorStop(0, 'rgba(224,235,237,.88)');
  fog.addColorStop(.52, 'rgba(202,218,221,.83)');
  fog.addColorStop(1, 'rgba(175,196,201,.86)');
  ctx.fillStyle = fog;
  ctx.fillRect(0, 0, innerWidth, innerHeight);
  if (noisePattern) {
    ctx.fillStyle = noisePattern;
    ctx.fillRect(0, 0, innerWidth, innerHeight);
  }
  ctx.save();
  ctx.globalCompositeOperation = 'destination-out';
  ctx.drawImage(mask, 0, 0, innerWidth, innerHeight);
  ctx.restore();
}

function isFingerWriting(landmarks: NormalizedLandmark[]): boolean {
  const indexTip = landmarks[8]!;
  const indexPip = landmarks[6]!;
  const otherTips = [landmarks[12]!, landmarks[16]!, landmarks[20]!];
  const indexExtended = indexTip.y < indexPip.y - 0.012;
  const indexIsLeading = otherTips.filter((tip) => indexTip.y + 0.018 < tip.y).length >= 2;
  return indexExtended && indexIsLeading;
}

function updateHand(result: HandLandmarkerResult): void {
  const landmarks = result.landmarks[0];
  if (!landmarks) {
    lastFinger = null;
    smoothFinger = null;
    cursor.style.opacity = '0';
    setState(handState, false, '未检测到手 · 伸出食指');
    return;
  }
  const tip = landmarks[8]!;
  const raw = { x: (1 - tip.x) * innerWidth, y: tip.y * innerHeight };
  smoothFinger = smoothFinger
    ? { x: smoothFinger.x * 0.55 + raw.x * 0.45, y: smoothFinger.y * 0.55 + raw.y * 0.45 }
    : raw;
  const writing = isFingerWriting(landmarks);
  cursor.style.opacity = '1';
  cursor.style.transform = `translate(${smoothFinger.x}px, ${smoothFinger.y}px)`;
  cursor.classList.toggle('drawing', writing);
  setState(handState, true, writing ? '正在书写 · 移动食指' : '已检测手部 · 让食指高于其他手指');
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
  const jawOpen = scores?.find((item) => item.categoryName === 'jawOpen')?.score ?? 0;
  if (!face) {
    puckerStartedAt = 0;
    setState(faceState, false, '未检测到面部 · 正对镜头');
    return;
  }
  const active = pucker > 0.56 && jawOpen > 0.06;
  const progress = puckerStartedAt ? Math.min(100, Math.round(((now - puckerStartedAt) / 750) * 100)) : 0;
  setState(faceState, true, active ? `保持哈气嘴形 · ${progress}%` : '已检测面部 · 噘嘴并微微张口');
  if (!active) {
    puckerStartedAt = 0;
    return;
  }
  if (!puckerStartedAt) puckerStartedAt = now;
  if (now - puckerStartedAt > 750 && now - lastBreathAt > 1800) {
    const upperLip = face[13]!;
    const lowerLip = face[14]!;
    const mouth = { x: (1 - (upperLip.x + lowerLip.x) / 2) * innerWidth, y: ((upperLip.y + lowerLip.y) / 2) * innerHeight };
    const faceWidth = distance(face[234]!, face[454]!) * innerWidth;
    const breathRadius = Math.max(80, Math.min(175, faceWidth * 0.46));
    addFog(mouth, breathRadius);
    addFog({ x: mouth.x - breathRadius * 0.38, y: mouth.y - breathRadius * 0.1 }, breathRadius * 0.55);
    addFog({ x: mouth.x + breathRadius * 0.34, y: mouth.y + breathRadius * 0.08 }, breathRadius * 0.5);
    breathPulse.style.left = `${mouth.x}px`;
    breathPulse.style.top = `${mouth.y}px`;
    breathPulse.classList.remove('play');
    void breathPulse.offsetWidth;
    breathPulse.classList.add('play');
    lastBreathAt = now;
    puckerStartedAt = 0;
    showToast('呼——镜面重新起雾了');
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
  render();
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
