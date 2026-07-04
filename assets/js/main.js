// Shared bootstrap and application lifecycle for the GANN stick-figure game.
'use strict';

const {canvas, chart, els} = window.GANN_DOM;
const cfg = window.GANN_CONFIG;
const ctx = canvas.getContext('2d');
const cctx = chart.getContext('2d');
const brainGenomeLength = (cfg.inputCount + 1) * cfg.hiddenCount + (cfg.hiddenCount + 1) * cfg.outputCount;
const traitStart = brainGenomeLength;
const genomeLength = brainGenomeLength + cfg.traitCount;

let seed = (Date.now() ^ 0x9e3779b9) >>> 0;
let rng = mulberry32(seed);
let population = [];
let generation = 1;
let episodeTime = 0;
let paused = false;
let stage = 'walk';
let bestEver = null;
let bestThisGen = null;
let history = [];
let cameraX = 0;
let generationJustEvolved = false;
let won = false;
let stagnation = makeStagnation();


function mulberry32(a){
  return function(){
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
function rand(min=0,max=1){return min + (max-min)*rng();}
function randn(){
  let u = 0, v = 0;
  while(u === 0) u = rng();
  while(v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function loop(){
  update();
  drawWorld();
  drawChart();
  updateUI();
  generationJustEvolved = false;
  requestAnimationFrame(loop);
}

function reset(seedMode=false){
  seed = (Date.now() ^ Math.floor(Math.random()*0xffffffff)) >>> 0;
  rng = mulberry32(seed);
  generation = 1; episodeTime = 0; stage = 'walk'; bestEver = null; history = []; cameraX = 0; won = false; stagnation = makeStagnation();
  initPopulation(seedMode);
}

function saveChampion(){
  if(!bestEver) return;
  const packed = {genome:Array.from(bestEver.genome), fitness:bestEver.fitness, x:bestEver.x, avgSpeed:bestEver.avgSpeed, stage, savedAt:new Date().toISOString()};
  localStorage.setItem(cfg.storageKeys[0], JSON.stringify(packed));
  els.save.textContent = 'Saved'; setTimeout(()=>els.save.textContent='Save champion',900);
}
function loadChampion(){
  const txt = cfg.storageKeys.map(key => localStorage.getItem(key)).find(Boolean);
  if(!txt){ els.load.textContent = 'No save'; setTimeout(()=>els.load.textContent='Load champion',900); return; }
  try{
    const packed = JSON.parse(txt);
    if(!packed.genome || packed.genome.length < 32) throw new Error('bad genome');
    const g = new Float32Array(genomeLength);
    g.set(new Float32Array(packed.genome).subarray(0, Math.min(packed.genome.length, genomeLength)));
    population = [];
    for(let i=0;i<cfg.popSize;i++) population.push(new Agent(i===0 ? cloneGenome(g) : mutate(g, 0.08, 0.22), i));
    bestEver = {genome:cloneGenome(g), fitness:packed.fitness||0, x:packed.x||0, avgSpeed:packed.avgSpeed||0, age:0, stage:packed.stage||stage};
    stage = packed.stage === 'run' ? 'run' : 'walk';
    generation = 1; episodeTime = 0; won = false; history=[]; stagnation = makeStagnation();
    els.load.textContent = 'Loaded'; setTimeout(()=>els.load.textContent='Load champion',900);
  }catch(e){
    els.load.textContent = 'Bad save'; setTimeout(()=>els.load.textContent='Load champion',900);
  }
}

els.toggle.addEventListener('click',()=>{paused=!paused;});
els.reset.addEventListener('click',()=>reset(false));
els.cold.addEventListener('click',()=>reset(true));
els.forceRun.addEventListener('click',()=>{
  stage = 'run'; won = false; episodeTime = 0; history=[]; stagnation = makeStagnation();
  const champ = bestEver ? bestEver.genome : makeRandomGenome(0.78);
  population=[]; for(let i=0;i<cfg.popSize;i++) population.push(new Agent(i===0 ? cloneGenome(champ) : mutate(champ, 0.16, 0.40), i));
});
els.save.addEventListener('click',saveChampion);
els.load.addEventListener('click',loadChampion);

canvas.addEventListener('click',(ev)=>{
  const leader = bestThisGen || population[0];
  if(!leader || !leader.alive) return;
  const rect = canvas.getBoundingClientRect();
  const x = ev.clientX - rect.left;
  const mid = rect.width * 0.5;
  const shove = clamp((x - mid) / mid, -1, 1);
  leader.vx += shove * 0.9;
  leader.av += shove * 1.3;
});
window.addEventListener('keydown',(ev)=>{
  if(ev.target && ['INPUT','SELECT','TEXTAREA'].includes(ev.target.tagName)) return;
  if(ev.code === 'Space'){ paused=!paused; ev.preventDefault(); }
  if(ev.key.toLowerCase() === 'r') reset(false);
  if(ev.key.toLowerCase() === 'f') els.speed.value = els.speed.value === '40' ? '1' : '40';
});

resize();
initPopulation(false);
loop();
