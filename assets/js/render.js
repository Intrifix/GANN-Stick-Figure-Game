// Canvas rendering for the world, agents, camera, and fitness chart.
'use strict';

function resize(){
  const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.floor(rect.width * dpr);
  canvas.height = Math.floor(rect.height * dpr);
  ctx.setTransform(dpr,0,0,dpr,0,0);
}
window.addEventListener('resize', resize);

function worldToScreen(x,y){
  const ground = canvas.clientHeight - 74;
  return {x:(x - cameraX) * cfg.scale + canvas.clientWidth * 0.28, y:ground - y * cfg.scale};
}
function drawLine(a,b,color,width=3,alpha=1){
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = 'round';
  ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke();
  ctx.globalAlpha = 1;
}
function drawAgent(a, color='#e5f4ff', alpha=1, width=3.2){
  if(!a) return;
  const feet = a.computeFeet ? a.computeFeet() : null;
  const hip = worldToScreen(a.x, a.y);
  const torsoLen = 0.72, neckLen = 0.13;
  const shoulderW = 0.23;
  const torsoAngle = a.angle;
  const chestW = {x: a.x + Math.sin(torsoAngle) * torsoLen, y: a.y + Math.cos(torsoAngle) * torsoLen};
  const neckW = {x: chestW.x + Math.sin(torsoAngle) * neckLen, y: chestW.y + Math.cos(torsoAngle) * neckLen};
  const chest = worldToScreen(chestW.x, chestW.y);
  const neck = worldToScreen(neckW.x, neckW.y);
  const head = worldToScreen(neckW.x + Math.sin(torsoAngle) * 0.08, neckW.y + Math.cos(torsoAngle) * 0.08);
  drawLine(hip, chest, color, width, alpha);
  drawLine(chest, neck, color, width, alpha);
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath(); ctx.arc(head.x, head.y - 7, 10, 0, Math.PI*2); ctx.stroke();
  ctx.globalAlpha = 1;

  function drawLeg(leg, hipColor){
    const knee = worldToScreen(leg.kneeX, leg.kneeY);
    const foot = worldToScreen(leg.x, Math.max(leg.y,0));
    drawLine(hip,knee,hipColor,width,alpha);
    drawLine(knee,foot,hipColor,width,alpha);
    ctx.globalAlpha = alpha;
    ctx.fillStyle = leg.contact ? '#a7f3d0' : hipColor;
    ctx.beginPath(); ctx.ellipse(foot.x + (leg.side === 'L' ? -3 : 3), foot.y + 2, 8, 3, 0, 0, Math.PI*2); ctx.fill();
    ctx.globalAlpha = 1;
  }
  if(feet){ drawLeg(feet.left, color); drawLeg(feet.right, color); }

  const lShoulder = {x: chestW.x - Math.cos(torsoAngle) * shoulderW, y: chestW.y + Math.sin(torsoAngle) * shoulderW};
  const rShoulder = {x: chestW.x + Math.cos(torsoAngle) * shoulderW, y: chestW.y - Math.sin(torsoAngle) * shoulderW};
  function arm(shoulder, ang, side){
    const a1 = torsoAngle + ang + (side === 'L' ? -0.12 : 0.12);
    const elbowW = {x: shoulder.x + Math.sin(a1) * 0.36, y: shoulder.y - Math.cos(a1) * 0.36};
    const handW = {x: elbowW.x + Math.sin(a1 + (side === 'L' ? 0.35 : -0.35)) * 0.32, y: elbowW.y - Math.cos(a1 + (side === 'L' ? 0.35 : -0.35)) * 0.32};
    const sp = worldToScreen(shoulder.x, shoulder.y), ep = worldToScreen(elbowW.x, elbowW.y), hp = worldToScreen(handW.x, handW.y);
    drawLine(sp, ep, color, width*0.72, alpha*0.85); drawLine(ep, hp, color, width*0.72, alpha*0.85);
  }
  arm(lShoulder, a.j ? a.j.ls : 0, 'L'); arm(rShoulder, a.j ? a.j.rs : 0, 'R');
}

function drawWorld(){
  const w = canvas.clientWidth, h = canvas.clientHeight;
  ctx.clearRect(0,0,w,h);
  const ground = h - 74;
  const grad = ctx.createLinearGradient(0,0,0,h);
  grad.addColorStop(0,'#0c1429'); grad.addColorStop(1,'#07101e');
  ctx.fillStyle = grad; ctx.fillRect(0,0,w,h);

  let camTarget = 0;
  const leader = bestThisGen || population[0];
  if(els.cameraMode.value === 'leader') camTarget = leader ? leader.x - 0.9 : 0;
  else if(els.cameraMode.value === 'best') camTarget = bestEver ? bestEver.x - 0.9 : 0;
  else camTarget = 0;
  cameraX = lerp(cameraX, camTarget, 0.08);

  // track grid
  ctx.strokeStyle = 'rgba(125,211,252,.13)'; ctx.lineWidth = 1;
  const startM = Math.floor(cameraX - w / cfg.scale);
  const endM = Math.ceil(cameraX + w / cfg.scale);
  for(let x=startM;x<=endM;x++){
    const sx = worldToScreen(x,0).x;
    if(sx < -30 || sx > w+30) continue;
    ctx.beginPath(); ctx.moveTo(sx,ground); ctx.lineTo(sx,ground+12); ctx.stroke();
    if(x % 5 === 0){
      ctx.fillStyle = 'rgba(220,235,255,.45)'; ctx.font = '12px ui-monospace, monospace'; ctx.fillText(`${x} m`, sx+4, ground+31);
      ctx.beginPath(); ctx.moveTo(sx,0); ctx.lineTo(sx,ground); ctx.strokeStyle='rgba(125,211,252,.05)'; ctx.stroke(); ctx.strokeStyle = 'rgba(125,211,252,.13)';
    }
  }
  ctx.strokeStyle = '#52658d'; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(0,ground); ctx.lineTo(w,ground); ctx.stroke();
  ctx.fillStyle = '#07101e'; ctx.fillRect(0,ground+3,w,h-ground);

  const goal = stage === 'walk' ? cfg.trackWinWalk : cfg.trackWinRun;
  const goalX = worldToScreen(goal,0).x;
  if(goalX > -80 && goalX < w + 80){
    ctx.strokeStyle = stage === 'walk' ? '#7dd3fc' : '#a7f3d0'; ctx.lineWidth = 2; ctx.setLineDash([6,5]);
    ctx.beginPath(); ctx.moveTo(goalX,20); ctx.lineTo(goalX,ground); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = stage === 'walk' ? '#7dd3fc' : '#a7f3d0'; ctx.font = 'bold 12px ui-sans-serif, system-ui';
    ctx.fillText(stage === 'walk' ? 'WALK UNLOCK' : 'RUN WIN', goalX+8, 36);
  }

  const sorted = [...population].sort((a,b)=>a.x-b.x);
  for(const a of sorted){
    if(!a.alive) continue;
    const alpha = a === bestThisGen ? 0 : 0.13;
    if(alpha) drawAgent(a, '#9fb0cc', alpha, 2.1);
  }
  if(bestThisGen){
    if(bestThisGen.trail.length > 2){
      ctx.strokeStyle = 'rgba(125,211,252,.33)'; ctx.lineWidth = 2; ctx.beginPath();
      bestThisGen.trail.forEach((p,i)=>{ const sp=worldToScreen(p.x,p.y); if(i===0)ctx.moveTo(sp.x,sp.y); else ctx.lineTo(sp.x,sp.y); }); ctx.stroke();
    }
    drawAgent(bestThisGen, won ? '#fef08a' : '#eaf7ff', 1, 4.2);
  }

  if(bestEver && els.cameraMode.value !== 'leader'){
    const ghost = new Agent(bestEver.genome, -1);
    ghost.x = bestEver.x; ghost.y = 1.05; ghost.age = bestEver.age || 0;
    drawAgent(ghost, '#a7f3d0', 0.38, 2.7);
  }

  ctx.fillStyle = 'rgba(7,12,24,.72)'; ctx.fillRect(16,18,Math.min(560,w-32),62);
  ctx.strokeStyle = 'rgba(125,211,252,.18)'; ctx.strokeRect(16,18,Math.min(560,w-32),62);
  ctx.fillStyle = '#ecf4ff'; ctx.font = 'bold 18px ui-sans-serif, system-ui';
  const title = won ? 'Run unlocked: stable fast locomotion achieved!' : stage === 'walk' ? 'Goal: evolve a balanced walking gait' : 'Goal: turn the walker into a runner';
  ctx.fillText(title, 30, 43);
  ctx.fillStyle = '#9fb0cc'; ctx.font = '13px ui-sans-serif, system-ui';
  const sub = `Target ${ms(targetSpeed())} · best now ${bestThisGen ? m(Math.max(0,bestThisGen.x)) : '0.0 m'} · average speed ${bestThisGen ? ms(bestThisGen.avgSpeed) : '0.00 m/s'}`;
  ctx.fillText(sub, 30, 64);
}

function drawChart(){
  const w = chart.width, h = chart.height;
  cctx.clearRect(0,0,w,h);
  cctx.fillStyle = '#081020'; cctx.fillRect(0,0,w,h);
  cctx.strokeStyle = 'rgba(255,255,255,.08)'; cctx.lineWidth = 1;
  for(let i=1;i<4;i++){ cctx.beginPath(); cctx.moveTo(0,h*i/4); cctx.lineTo(w,h*i/4); cctx.stroke(); }
  if(history.length < 2) return;
  const maxFit = Math.max(...history.map(v=>Math.max(v.best, v.avg)), 10);
  function plot(key, color){
    cctx.strokeStyle = color; cctx.lineWidth = 2; cctx.beginPath();
    history.forEach((p,i)=>{
      const x = i / Math.max(1, history.length-1) * (w-8) + 4;
      const y = h - 5 - clamp(p[key] / maxFit, 0, 1) * (h-13);
      if(i===0) cctx.moveTo(x,y); else cctx.lineTo(x,y);
    }); cctx.stroke();
  }
  plot('best','#7dd3fc'); plot('avg','#a7f3d0');
  // mark stage transition
  cctx.strokeStyle = 'rgba(251,191,36,.5)'; cctx.setLineDash([3,4]);
  for(let i=1;i<history.length;i++) if(history[i-1].stage !== history[i].stage){
    const x = i / Math.max(1, history.length-1) * (w-8) + 4;
    cctx.beginPath(); cctx.moveTo(x,4); cctx.lineTo(x,h-4); cctx.stroke();
  }
  cctx.setLineDash([]);
}
