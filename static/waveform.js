const canvas = document.getElementById("bg-waveform");
let ctx = null;
let animationId = null;
const BAR_COUNT = 60;
const SMOOTHING = 0.1;
const UPDATE_EVERY = 20; // frames
let samples = new Array(BAR_COUNT).fill(0);
let targets = new Array(BAR_COUNT).fill(0);
let frame = 0;

function resize() {
  if (!canvas) return;
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
}

function updateSamples() {
  if (frame % UPDATE_EVERY === 0) {
    targets = targets.map(() => Math.random());
  }
  samples = samples.map((v, i) => v + (targets[i] - v) * SMOOTHING);
  frame++;
}

function draw() {
  if (!ctx) return;
  resize();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const color = getComputedStyle(document.documentElement).getPropertyValue("--accent") || "#fff";
  ctx.fillStyle = color.trim();
  ctx.globalAlpha = 0.35;
  const barWidth = canvas.width / BAR_COUNT;
  updateSamples();
  samples.forEach((v, i) => {
    const x = i * barWidth;
    const h = v * canvas.height;
    ctx.fillRect(x, canvas.height - h, barWidth * 0.7, h);
  });
  animationId = requestAnimationFrame(draw);
}

export function startWaveform() {
  if (!canvas) return;
  ctx = canvas.getContext("2d");
  if (animationId) cancelAnimationFrame(animationId);
  samples.fill(0);
  targets.fill(0);
  frame = 0;
  draw();
}

export function stopWaveform() {
  if (animationId) cancelAnimationFrame(animationId);
  animationId = null;
  if (ctx && canvas) ctx.clearRect(0, 0, canvas.width, canvas.height);
}

window.addEventListener("resize", resize);

