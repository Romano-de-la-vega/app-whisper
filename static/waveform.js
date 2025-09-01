const canvas = document.getElementById("bg-waveform");
let ctx = null;
let animationId = null;
const BAR_COUNT = 60;
const BASE_HEIGHT = 0.35; // base bar height ratio
const AMPLITUDE = 0.15;   // maximum deviation from base
const SPEED = 0.02;       // animation speed

let samples = new Array(BAR_COUNT).fill(0);
const offsets = Array.from({ length: BAR_COUNT }, (_, i) => i * 0.3);
let time = 0;

function resize() {
  if (!canvas) return;
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
}

function updateSamples() {
  time += SPEED;
  samples = offsets.map((o) => BASE_HEIGHT + Math.sin(time + o) * AMPLITUDE);
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
  time = 0;
  draw();
}

export function stopWaveform() {
  if (animationId) cancelAnimationFrame(animationId);
  animationId = null;
  if (ctx && canvas) ctx.clearRect(0, 0, canvas.width, canvas.height);
}

window.addEventListener("resize", resize);

