(function(){
  const canvas = document.getElementById('bg-particles');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const COUNT = 80;
  let particles = [];
  let frameId = null;

  function resize(){
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  }

  function createParticles(){
    particles = Array.from({length: COUNT}, () => ({
      x: Math.random()*canvas.width,
      y: Math.random()*canvas.height,
      vx: (Math.random()-0.5)*0.5,
      vy: (Math.random()-0.5)*0.5,
      r: 1 + Math.random()*2
    }));
  }

  function step(){
    ctx.clearRect(0,0,canvas.width,canvas.height);
    particles.forEach(p => {
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
      if (p.y < 0 || p.y > canvas.height) p.vy *= -1;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI*2);
      ctx.fillStyle = 'rgba(255,255,255,0.7)';
      ctx.fill();
    });
    frameId = requestAnimationFrame(step);
  }

  function start(){
    if (frameId) return;
    resize();
    createParticles();
    step();
  }

  function stop(){
    if (frameId) cancelAnimationFrame(frameId);
    frameId = null;
    ctx.clearRect(0,0,canvas.width,canvas.height);
  }

  window.Particles = { start, stop };
  window.addEventListener('resize', () => {
    if (!frameId) return; // only resize when running
    resize();
  });
})();
