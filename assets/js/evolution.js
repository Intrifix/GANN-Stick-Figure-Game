// Genetic algorithm, population management, stagnation recovery, and simulation ticking.
'use strict';


function makeStagnation(){
  return {
    bestProgress: -Infinity,
    stale: 0,
    escapes: 0,
    lastEscapeGen: -999,
    lastDiversity: 0,
    lastProgress: 0,
    lastAction: 'normal',
    lastReason: ''
  };
}

function makeRandomGenome(scale=0.8){
  const g = new Float32Array(genomeLength);
  for(let i=0;i<g.length;i++) g[i] = randn() * scale;
  return g;
}
function cloneGenome(g){ return new Float32Array(g); }
function mutate(g, rate, power){
  const out = cloneGenome(g);
  for(let i=0;i<out.length;i++){
    if(rng() < rate) out[i] += randn() * power;
    if(rng() < rate * 0.025) out[i] *= rand(0.25, 1.75);
    out[i] = clamp(out[i], -6, 6);
  }
  return out;
}
function crossover(a,b){
  const g = new Float32Array(genomeLength);
  const cut1 = Math.floor(rand(0, genomeLength));
  const cut2 = Math.floor(rand(cut1, genomeLength));
  for(let i=0;i<g.length;i++){
    if(i>=cut1 && i<cut2) g[i] = b[i];
    else g[i] = rng() < 0.5 ? a[i] : b[i];
    if(rng() < 0.08) g[i] = (a[i] + b[i]) * 0.5;
  }
  return g;
}
function genomeIndexHidden(j, i){ return j * (cfg.inputCount + 1) + 1 + i; }
function genomeIndexHiddenBias(j){ return j * (cfg.inputCount + 1); }
function genomeIndexOutput(o, j){ return (cfg.inputCount + 1) * cfg.hiddenCount + o * (cfg.hiddenCount + 1) + 1 + j; }
function genomeIndexOutputBias(o){ return (cfg.inputCount + 1) * cfg.hiddenCount + o * (cfg.hiddenCount + 1); }
function swapHiddenInputs(g, a, b){
  for(let j=0;j<cfg.hiddenCount;j++){
    const ia = genomeIndexHidden(j, a), ib = genomeIndexHidden(j, b);
    const t = g[ia]; g[ia] = g[ib]; g[ib] = t;
  }
}
function swapOutputRows(g, a, b){
  const ba = genomeIndexOutputBias(a), bb = genomeIndexOutputBias(b);
  let t = g[ba]; g[ba] = g[bb]; g[bb] = t;
  for(let j=0;j<cfg.hiddenCount;j++){
    const ia = genomeIndexOutput(a, j), ib = genomeIndexOutput(b, j);
    t = g[ia]; g[ia] = g[ib]; g[ib] = t;
  }
}
function mirroredGenome(genome){
  const g = cloneGenome(genome);
  // Swap left/right sensor paths and motor outputs. These mirror mutants stop a
  // one-legged local optimum from monopolizing the gene pool: the GA can cross a
  // left-leg specialist with its right-leg mirror and discover alternating gaits.
  swapHiddenInputs(g, 7, 8);   // left/right contacts
  swapHiddenInputs(g, 9, 10);  // left/right foot positions
  swapHiddenInputs(g, 11, 12); // left/right hip states
  swapHiddenInputs(g, 13, 14); // left/right knee states
  swapHiddenInputs(g, 18, 19); // left/right foot heights
  swapHiddenInputs(g, 20, 21); // left/right foot rel velocities
  swapHiddenInputs(g, 22, 23); // left/right stance timers
  swapHiddenInputs(g, 24, 25); // left/right contact-time ratios
  swapHiddenInputs(g, 26, 27); // left/right touch counts
  swapHiddenInputs(g, 28, 29); // left/right good steps
  swapHiddenInputs(g, 32, 33); // left/right ankles
  swapHiddenInputs(g, 35, 38); // previous left/right hip commands
  swapHiddenInputs(g, 36, 39); // previous left/right knee commands
  swapHiddenInputs(g, 37, 40); // previous left/right ankle commands
  swapHiddenInputs(g, 41, 42); // previous arm commands
  swapHiddenInputs(g, 45, 46); // previous swing commands
  swapHiddenInputs(g, 47, 48); // previous stance stiffness commands
  swapOutputRows(g, 0, 3); swapOutputRows(g, 1, 4); swapOutputRows(g, 2, 5);
  swapOutputRows(g, 6, 7); swapOutputRows(g, 10, 11); swapOutputRows(g, 12, 13);
  return g;
}

function makeStarterGenome(kind='walk'){
  const g = new Float32Array(genomeLength);
  // Hidden nodes mirror oscillator and feedback inputs; output layer combines them into a rough gait.
  const H = {
    sinP:0, sinN:1, cosP:2, cosN:3, slow:4, fast:5, leanF:6, leanB:7,
    contactL:8, contactR:9, speedLow:10, speedHigh:11
  };
  function h(j, input, weight, bias=0){ g[genomeIndexHiddenBias(j)] = bias; g[genomeIndexHidden(j,input)] = weight; }
  h(H.sinP, 0, kind === 'run' ? 3.0 : 2.65);
  h(H.sinN, 0, kind === 'run' ? -3.0 : -2.65);
  h(H.cosP, 1, 2.4);
  h(H.cosN, 1, -2.4);
  h(H.slow, 16, -1.4, 0.2);
  h(H.fast, 16, 1.4, -0.1);
  h(H.leanF, 4, -2.0);
  h(H.leanB, 4, 2.0);
  h(H.contactL, 7, 2.2, -0.7);
  h(H.contactR, 8, 2.2, -0.7);
  h(H.speedLow, 3, 1.7, -0.35);
  h(H.speedHigh, 2, 1.2, -0.7);
  function o(out, hidden, weight){ g[genomeIndexOutput(out,hidden)] += weight; }
  function ob(out, bias){ g[genomeIndexOutputBias(out)] = bias; }
  const runBoost = kind === 'run' ? 1.22 : 1.0;
  // left hip, left knee, left ankle
  o(0,H.sinP,1.35*runBoost); o(0,H.speedLow,0.18); ob(0,0.02);
  o(1,H.sinP,0.95); o(1,H.cosN,0.28); ob(1,-0.38);
  o(2,H.sinN,-0.18); o(2,H.contactL,-0.38); ob(2,-0.05);
  // right hip, right knee, right ankle
  o(3,H.sinN,1.35*runBoost); o(3,H.speedLow,0.18); ob(3,0.02);
  o(4,H.sinN,0.95); o(4,H.cosP,0.28); ob(4,-0.38);
  o(5,H.sinP,-0.18); o(5,H.contactR,-0.38); ob(5,-0.05);
  // arms counter-swing
  o(6,H.sinN,1.25); ob(6,0);
  o(7,H.sinP,1.25); ob(7,0);
  // lean and muscle
  o(8,H.leanF,0.55); o(8,H.speedLow,0.20); o(8,H.fast,0.10); ob(8, kind === 'run' ? 0.18 : 0.06);
  o(9,H.fast,0.35); o(9,H.speedLow,0.24); ob(9, kind === 'run' ? 0.28 : 0.08);
  g[traitStart + 9] = 1.75; // seeded gaits deliberately get a strong clock; cold-start genomes must evolve it.
  g[traitStart + 10] = kind === 'run' ? 0.35 : 0.0;
  g[traitStart + 11] = 0.25;
  g[traitStart + 12] = 0.15;
  return g;
}

function initPopulation(seeded=false){
  population = [];
  const baseWalk = makeStarterGenome('walk');
  const baseRun = makeStarterGenome('run');
  for(let i=0;i<cfg.popSize;i++){
    let g;
    if(seeded && i < cfg.popSize * 0.42) g = mutate(baseWalk, 0.22, 0.42);
    else if(seeded && i < cfg.popSize * 0.54) g = mutate(baseRun, 0.26, 0.50);
    else g = makeRandomGenome(0.72);
    population.push(new Agent(g, i));
  }
  bestThisGen = population[0];
}

function tournament(sorted){
  let best = sorted[Math.floor(rand(0, sorted.length))];
  for(let i=0;i<3;i++){
    const c = sorted[Math.floor(Math.pow(rng(), 1.7) * sorted.length)];
    if(c.fitness > best.fitness) best = c;
  }
  return best;
}

function gaitStats(a){
  const left = a && (a.leftGoodSteps || 0);
  const right = a && (a.rightGoodSteps || 0);
  const alt = a && (a.altGoodSteps || 0);
  const maxSide = Math.max(left, right);
  const balance = maxSide > 0 ? Math.min(left, right) / maxSide : 0;
  const leftTouches = a && (a.leftTouchCount || 0);
  const rightTouches = a && (a.rightTouchCount || 0);
  const maxTouches = Math.max(leftTouches, rightTouches);
  const touchBalance = maxTouches > 0 ? Math.min(leftTouches, rightTouches) / maxTouches : 0;
  const maxContactT = Math.max(a && (a.leftContactTime || 0), a && (a.rightContactTime || 0));
  const contactBalance = maxContactT > 0 ? Math.min(a && (a.leftContactTime || 0), a && (a.rightContactTime || 0)) / maxContactT : 0;
  return {left, right, alt, balance, touchBalance, contactBalance};
}
function validBipedGait(a, mode){
  const s = gaitStats(a);
  const contactRatio = (a && a.age) ? (a.loadedContactTime || 0) / Math.max(0.1, a.age) : 0;
  const doubleRatio = (a && a.age) ? (a.doubleContactTime || 0) / Math.max(0.1, a.age) : 0;
  if(mode === 'run') return s.left >= 4 && s.right >= 4 && s.alt >= 6 && s.balance >= 0.45 && s.touchBalance >= 0.50 && s.contactBalance >= 0.42 && contactRatio > 0.12 && doubleRatio < 0.42;
  return s.left >= 3 && s.right >= 3 && s.alt >= 4 && s.balance >= 0.45 && s.touchBalance >= 0.45 && s.contactBalance >= 0.38 && contactRatio > 0.16 && doubleRatio < 0.70;
}

function behaviorVector(a){
  const s = gaitStats(a);
  const age = Math.max(0.1, a.age || 0);
  const goal = stage === 'run' ? cfg.trackWinRun : cfg.trackWinWalk;
  return [
    clamp((a.x || 0) / goal, -0.2, 1.6),
    clamp((a.avgSpeed || 0) / Math.max(0.1, targetSpeed()), -0.5, 2.0),
    clamp(s.left / 8, 0, 2),
    clamp(s.right / 8, 0, 2),
    clamp(s.alt / 8, 0, 2),
    clamp(s.balance, 0, 1),
    clamp(s.touchBalance, 0, 1),
    clamp(s.contactBalance, 0, 1),
    clamp((a.doubleContactTime || 0) / age, 0, 1.4),
    clamp((a.airTime || 0) / age, 0, 1.4)
  ];
}
function behaviorDistanceVec(a,b){
  let d = 0;
  for(let i=0;i<a.length;i++){ const x = a[i] - b[i]; d += x*x; }
  return Math.sqrt(d / a.length);
}
function assignNovelty(pop){
  const vectors = pop.map(behaviorVector);
  for(let i=0;i<pop.length;i++){
    const distances = [];
    for(let j=0;j<pop.length;j++) if(i !== j) distances.push(behaviorDistanceVec(vectors[i], vectors[j]));
    distances.sort((a,b)=>a-b);
    const k = Math.min(5, distances.length);
    let score = 0;
    for(let n=0;n<k;n++) score += distances[n];
    pop[i].novelty = k ? score / k : 0;
  }
}
function populationDiversity(pop){
  if(pop.length < 2) return 0;
  const sample = Math.min(72, genomeLength);
  const stride = Math.max(1, Math.floor(genomeLength / sample));
  let total = 0, pairs = 0;
  const n = Math.min(pop.length, 22);
  for(let i=0;i<n;i++){
    for(let j=i+1;j<n;j++){
      let d = 0, c = 0;
      for(let k=0;k<genomeLength;k+=stride){
        d += Math.abs((pop[i].genome[k] || 0) - (pop[j].genome[k] || 0)); c++;
      }
      total += d / Math.max(1,c); pairs++;
    }
  }
  return pairs ? total / pairs : 0;
}
function progressMetric(a){
  if(!a) return -Infinity;
  const s = gaitStats(a);
  const age = Math.max(0.1, a.age || 0);
  const dist = Math.max(0, a.x || 0);
  const minSteps = Math.min(s.left, s.right);
  const maxSteps = Math.max(s.left, s.right);
  const doubleRatio = (a.doubleContactTime || 0) / age;
  const hoverRatio = (a.airTime || 0) / age;
  const oneLegTrap = Math.max(0, maxSteps - Math.max(1, minSteps) * 2) * 0.75 + Math.max(0, 0.42 - s.balance) * Math.max(0, dist - 2.5) * 0.55;
  const gluedTrap = Math.max(0, doubleRatio - (stage === 'run' ? 0.42 : 0.66)) * Math.max(0, dist - 1.0) * 2.0;
  const hoverTrap = Math.max(0, hoverRatio - (stage === 'run' ? 0.50 : 0.30)) * Math.max(0, dist - 1.0) * 2.0;
  return dist * 1.25 + Math.max(0, a.avgSpeed || 0) * 2.4 + minSteps * 2.0 + (s.alt || 0) * 1.4 + s.balance * 3.0 + s.touchBalance * 1.6 - oneLegTrap - gluedTrap - hoverTrap;
}
function analyzeStagnation(sorted, avgFitness, diversity){
  const best = sorted[0];
  const progress = progressMetric(best);
  stagnation.lastProgress = progress;
  stagnation.lastDiversity = diversity;
  const improved = progress > stagnation.bestProgress + (stage === 'run' ? 0.55 : 0.38);
  if(improved){
    stagnation.bestProgress = progress;
    stagnation.stale = 0;
    if(stagnation.lastAction !== 'normal') stagnation.lastAction = 'recovering';
  } else {
    stagnation.stale++;
  }
  const s = gaitStats(best);
  const age = Math.max(0.1, best.age || 0);
  const dist = Math.max(0, best.x || 0);
  const oneLegTrap = dist > 2.6 && Math.max(s.left, s.right) >= 2 && s.balance < 0.34;
  const gluedTrap = dist > 1.4 && (best.doubleContactTime || 0) / age > (stage === 'run' ? 0.48 : 0.76);
  const lowDiversity = diversity < 0.47;
  const noRecentEscape = generation - stagnation.lastEscapeGen > 4;
  let reason = '';
  if(stagnation.stale >= 8) reason = 'stale';
  if(stagnation.stale >= 5 && oneLegTrap) reason = 'one-leg trap';
  if(stagnation.stale >= 5 && gluedTrap) reason = 'glued-feet trap';
  if(stagnation.stale >= 6 && lowDiversity) reason = 'low diversity';
  const triggered = Boolean(reason && noRecentEscape);
  if(triggered){
    stagnation.escapes++;
    stagnation.lastEscapeGen = generation;
    stagnation.lastAction = reason;
    stagnation.lastReason = reason;
    stagnation.stale = Math.floor(stagnation.stale * 0.45);
  }
  return {triggered, reason, progress, oneLegTrap, gluedTrap, lowDiversity};
}

function evolve(){
  const target = targetSpeed();
  for(const a of population){ a.fitness = a.score(target, true); a.avgSpeed = a.age > 0 ? a.x / a.age : 0; }
  assignNovelty(population);
  population.sort((a,b)=>b.fitness-a.fitness);
  const avg = population.reduce((sum,a)=>sum+a.fitness,0) / population.length;
  const diversity = populationDiversity(population);
  const escape = analyzeStagnation(population, avg, diversity);

  // When the population has been stale for a while, give novel behaviors a
  // small selection nudge even before a full partial restart is triggered.
  if(stagnation.stale >= 3){
    const noveltyBoost = stage === 'run' ? 3.2 : 2.4;
    for(const a of population) a.fitness += (a.novelty || 0) * noveltyBoost;
    population.sort((a,b)=>b.fitness-a.fitness);
  }

  bestThisGen = population[0];
  if(!bestEver || bestThisGen.fitness > bestEver.fitness || (stage === 'run' && bestThisGen.x > bestEver.x && bestThisGen.avgSpeed > 1.5)){
    bestEver = snapshotAgent(bestThisGen);
  }
  const adjustedAvg = population.reduce((s,a)=>s+a.fitness,0) / population.length;
  history.push({best: bestThisGen.fitness, avg: adjustedAvg, dist: bestThisGen.x, speed: bestThisGen.avgSpeed, stage});
  if(history.length > 170) history.shift();

  const walkUnlocked = bestThisGen.x > cfg.trackWinWalk && bestThisGen.avgSpeed > 0.85 && validBipedGait(bestThisGen, 'walk') || bestEver && bestEver.x > cfg.trackWinWalk && bestEver.avgSpeed > 0.85 && validBipedGait(bestEver, 'walk');
  if(stage === 'walk' && walkUnlocked){
    stage = 'run';
    stagnation = makeStagnation();
    // Inject the best walker into a run-biased gene pool instead of starting over.
    const champion = cloneGenome(bestThisGen.genome);
    population = [];
    population.push(new Agent(champion, 0, bestThisGen.fitness));
    for(let i=1;i<cfg.popSize;i++){
      let child = cloneGenome(champion);
      if(i % 5 === 0) child = crossover(champion, makeRandomGenome(0.52));
      if(i % 7 === 0) child = crossover(child, mirroredGenome(champion));
      population.push(new Agent(mutate(child, 0.18, 0.44), i, bestThisGen.fitness));
    }
    bestEver = snapshotAgent(population[0]);
    generation++;
    episodeTime = 0;
    return;
  }
  if(stage === 'run' && !won && bestThisGen.x > cfg.trackWinRun && bestThisGen.avgSpeed > 2.45 && validBipedGait(bestThisGen, 'run')){
    won = true;
  }

  const next = [];
  const mutationRate = Number(els.mutation.value) / 100;
  const noveltyRanked = [...population].sort((a,b)=>(b.novelty||0)-(a.novelty||0));
  const keepElites = escape.triggered ? Math.min(2, cfg.eliteCount) : cfg.eliteCount;
  for(let i=0;i<keepElites;i++){
    next.push(new Agent(cloneGenome(population[i].genome), i, population[i].fitness));
  }

  const immigrantCount = escape.triggered ? Math.ceil(cfg.popSize * 0.25) : 0;
  const hyperCount = escape.triggered ? Math.ceil(cfg.popSize * 0.22) : 0;
  const mirrorCount = escape.triggered ? Math.ceil(cfg.popSize * 0.18) : 6;

  for(let i=keepElites;i<cfg.popSize;i++){
    let child;
    const power = stage === 'run' ? 0.26 : 0.21;
    const slotFromEnd = cfg.popSize - i;
    if(escape.triggered && slotFromEnd <= immigrantCount){
      // Random immigrants are the direct escape hatch: about 25% of the pool is
      // rebuilt from scratch so the population cannot stay genetically trapped.
      child = makeRandomGenome(stage === 'run' ? 0.90 : 0.80);
    } else if(escape.triggered && i < keepElites + hyperCount){
      // Hypermutate a mix of elites and behaviorally novel agents.
      const p = (i % 2 === 0) ? population[i % Math.min(8, population.length)] : noveltyRanked[i % Math.min(10, noveltyRanked.length)];
      let base = cloneGenome(p.genome);
      if(i % 3 === 0) base = crossover(base, mirroredGenome(base));
      if(i % 5 === 0) base = crossover(base, makeRandomGenome(0.70));
      child = mutate(base, Math.max(mutationRate * 2.2, 0.15), stage === 'run' ? 0.62 : 0.54);
    } else if(i < keepElites + hyperCount + mirrorCount){
      const p = population[(i - keepElites) % Math.min(cfg.eliteCount + 2, population.length)];
      child = crossover(p.genome, mirroredGenome(p.genome));
      child = mutate(child, Math.max(mutationRate, 0.080), power * 1.45);
    } else if(i > cfg.popSize - 6 && rng() < 0.70){
      child = makeRandomGenome(stage === 'run' ? 0.78 : 0.72);
    } else {
      const useNovel = stagnation.stale >= 3 && rng() < 0.28;
      const p1 = useNovel ? noveltyRanked[Math.floor(rand(0, Math.min(12, noveltyRanked.length)))] : tournament(population);
      const p2 = tournament(population);
      child = crossover(p1.genome, p2.genome);
      child = mutate(child, mutationRate * (stagnation.stale >= 3 ? 1.25 : 1), power);
    }
    next.push(new Agent(child, i));
  }
  population = next;
  generation++;
  episodeTime = 0;
  generationJustEvolved = true;
}
function targetSpeed(){ return stage === 'run' ? cfg.runTarget : cfg.walkTarget; }
function timeLimit(){ return stage === 'run' ? cfg.runTime : cfg.walkTime; }
function snapshotAgent(a){
  return {
    genome: cloneGenome(a.genome), fitness: a.fitness, x: a.x, avgSpeed: a.avgSpeed, age: a.age,
    goodSteps: a.goodSteps || 0, leftGoodSteps: a.leftGoodSteps || 0, rightGoodSteps: a.rightGoodSteps || 0,
    altGoodSteps: a.altGoodSteps || 0, sameFootGoodSteps: a.sameFootGoodSteps || 0,
    leftTouchCount: a.leftTouchCount || 0, rightTouchCount: a.rightTouchCount || 0,
    leftContactTime: a.leftContactTime || 0, rightContactTime: a.rightContactTime || 0,
    loadedContactTime: a.loadedContactTime || 0, driveContactTime: a.driveContactTime || 0,
    trail: a.trail ? a.trail.map(p => ({x:p.x,y:p.y})) : [], stage
  };
}

function update(){
  if(paused) return;
  let steps = Number(els.speed.value);
  for(let s=0;s<steps;s++){
    const target = targetSpeed();
    episodeTime += cfg.dt;
    let alive = 0;
    let leader = null;
    for(const a of population){
      a.step(cfg.dt, target);
      if(a.alive) alive++;
      if(!leader || a.fitness > leader.fitness) leader = a;
    }
    bestThisGen = leader || population[0];
    if(episodeTime >= timeLimit() || alive === 0) evolve();
  }
}
