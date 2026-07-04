// Agent physiology, neural network inference, and simulation scoring.
'use strict';

class Brain {
  constructor(genome){ this.g = genome; }
  think(input){
    const h = new Float32Array(cfg.hiddenCount);
    let k = 0;
    for(let j=0;j<cfg.hiddenCount;j++){
      let sum = this.g[k++];
      for(let i=0;i<cfg.inputCount;i++) sum += input[i] * this.g[k++];
      h[j] = tanh(sum);
    }
    const out = new Float32Array(cfg.outputCount);
    for(let o=0;o<cfg.outputCount;o++){
      let sum = this.g[k++];
      for(let j=0;j<cfg.hiddenCount;j++) sum += h[j] * this.g[k++];
      out[o] = tanh(sum);
    }
    return out;
  }
}

function traitGene(genome, index){
  return tanh(Number(genome[traitStart + index] || 0));
}
function traitRange(genome, index, center, spread, lo, hi){
  return clamp(center + traitGene(genome, index) * spread, lo, hi);
}
function decodeTraits(genome){
  return {
    sweepGain: traitRange(genome, 0, 1.00, 0.38, 0.58, 1.42),
    speedDamp: traitRange(genome, 1, 0.16, 0.08, 0.07, 0.27),
    sweepBias: traitRange(genome, 2, 0.00, 0.42, -0.50, 0.50),
    sweepForce: traitRange(genome, 3, 1.00, 0.32, 0.58, 1.38),
    behindDrive: traitRange(genome, 4, 0.92, 0.30, 0.52, 1.30),
    extensionDrive: traitRange(genome, 5, 0.23, 0.14, 0.06, 0.44),
    ankleDrive: traitRange(genome, 6, 0.12, 0.09, 0.02, 0.25),
    airDragX: traitRange(genome, 7, 0.085, 0.055, 0.035, 0.165),
    accelTolerance: traitRange(genome, 8, 14.0, 4.2, 8.5, 19.0),
    clockInput: traitRange(genome, 9, 0.25, 0.75, 0.0, 1.15),
    clockRate: traitRange(genome, 10, 1.00, 0.55, 0.35, 1.75),
    reflexGain: traitRange(genome, 11, 0.55, 0.35, 0.10, 1.05),
    motorSmoothing: traitRange(genome, 12, 1.00, 0.45, 0.45, 1.55)
  };
}

class Agent {
  constructor(genome, id, parentFitness=0){
    this.genome = genome;
    this.brain = new Brain(genome);
    this.traits = decodeTraits(genome);
    this.id = id;
    this.parentFitness = parentFitness;
    this.reset();
  }
  reset(){
    this.x = rand(-0.035, 0.035); this.y = rand(0.94, 1.12); this.vx = rand(-0.04, 0.04); this.vy = 0;
    this.angle = rand(-0.16,0.16); this.av = rand(-0.18,0.18);
    this.clockOffset = rand(0, Math.PI * 2);
    this.clockWobble = rand(0.82, 1.18);
    this.age = 0; this.alive = true; this.fallen = false;
    this.energy = 0; this.airTime = 0; this.groundTime = 0; this.steps = 0;
    this.accelPenalty = 0; this.jitterPenalty = 0; this.skatePenalty = 0; this.hoverPenalty = 0; this.badThrustPenalty = 0; this.legDominancePenalty = 0;
    this.airStreak = 0; this.loadedContactTime = 0; this.driveContactTime = 0; this.goodSteps = 0;
    this.leftGoodSteps = 0; this.rightGoodSteps = 0; this.altGoodSteps = 0; this.sameFootGoodSteps = 0;
    this.leftTouchCount = 0; this.rightTouchCount = 0; this.leftContactTime = 0; this.rightContactTime = 0;
    this.singleContactTime = 0; this.doubleContactTime = 0; this.gluedFeetPenalty = 0;
    this.lastGoodFoot = null; this.bestDualStepMin = 0;
    this.lPlantX = null; this.rPlantX = null; this.lStanceTime = 0; this.rStanceTime = 0;
    this.lTouchRel = 0; this.rTouchRel = 0; this.lMaxTravel = 0; this.rMaxTravel = 0;
    this.maxX = this.x; this.minY = this.y; this.fitness = 0; this.avgSpeed = 0;
    this.lastContactMask = 0;
    this.j = {
      lh: rand(-0.48,0.48), lk: rand(0.08,1.10), la: rand(-0.35,0.35),
      rh: rand(-0.48,0.48), rk: rand(0.08,1.10), ra: rand(-0.35,0.35),
      ls: rand(-0.75,0.75), rs: rand(-0.75,0.75), lean: rand(-0.20,0.20), muscle: rand(0.12,0.55),
      lswing: rand(0.05,0.45), rswing: rand(0.05,0.45), lstiff: rand(0.15,0.65), rstiff: rand(0.15,0.65)
    };
    this.lastOut = new Float32Array(cfg.outputCount);
    const f = this.computeFeet();
    this.prevRelL = f.left.relX; this.prevRelR = f.right.relX;
    this.prevRelVelL = 0; this.prevRelVelR = 0;
    this.prevLX = f.left.x; this.prevRX = f.right.x;
    this.prevVx = this.vx; this.prevVy = this.vy;
    this.trail = [];
  }
  inputVector(targetSpeed){
    const traits = this.traits;
    const baseRate = stage === 'run' ? 5.2 : 3.6;
    const phase = this.age * baseRate * traits.clockRate * this.clockWobble + this.clockOffset;
    const clockAmp = traits.clockInput;
    const feet = this.computeFeet();
    const lc = feet.left.contact ? 1 : 0;
    const rc = feet.right.contact ? 1 : 0;
    const age = Math.max(0.1, this.age);
    const out = new Float32Array(cfg.inputCount);
    let k = 0;
    out[k++] = Math.sin(phase) * clockAmp;
    out[k++] = Math.cos(phase) * clockAmp;
    out[k++] = clamp(this.vx / Math.max(0.1,targetSpeed), -2, 2);
    out[k++] = clamp((targetSpeed - this.vx) / Math.max(0.1,targetSpeed), -2, 2);
    out[k++] = clamp(this.angle, -1.5, 1.5);
    out[k++] = clamp(this.av, -3, 3);
    out[k++] = clamp((this.y - 1.03) * 2, -1, 1);
    out[k++] = lc; out[k++] = rc;
    out[k++] = clamp(feet.left.relX, -1, 1);
    out[k++] = clamp(feet.right.relX, -1, 1);
    out[k++] = clamp(this.j.lh, -1.5, 1.5);
    out[k++] = clamp(this.j.rh, -1.5, 1.5);
    out[k++] = clamp(this.j.lk - 0.45, -1, 1);
    out[k++] = clamp(this.j.rk - 0.45, -1, 1);
    out[k++] = clamp(this.energy / 500, 0, 2);
    out[k++] = stage === 'run' ? 1 : -1;
    out[k++] = 1;
    // Extra leg-state intelligence: foot height, foot sweep velocity, stance
    // timers, contact history, step history, vertical motion, ankles and muscle.
    out[k++] = clamp(feet.left.y * 3, -1, 1);
    out[k++] = clamp(feet.right.y * 3, -1, 1);
    out[k++] = clamp(this.prevRelVelL / 4, -2, 2);
    out[k++] = clamp(this.prevRelVelR / 4, -2, 2);
    out[k++] = clamp(this.lStanceTime * 2.2, 0, 2);
    out[k++] = clamp(this.rStanceTime * 2.2, 0, 2);
    out[k++] = clamp(this.leftContactTime / age, 0, 1.5);
    out[k++] = clamp(this.rightContactTime / age, 0, 1.5);
    out[k++] = clamp(this.leftTouchCount / 10, 0, 2);
    out[k++] = clamp(this.rightTouchCount / 10, 0, 2);
    out[k++] = clamp(this.leftGoodSteps / 8, 0, 2);
    out[k++] = clamp(this.rightGoodSteps / 8, 0, 2);
    out[k++] = clamp(this.altGoodSteps / 8, 0, 2);
    out[k++] = clamp(this.vy / 4, -2, 2);
    out[k++] = clamp(this.j.la, -1.5, 1.5);
    out[k++] = clamp(this.j.ra, -1.5, 1.5);
    out[k++] = clamp(this.j.muscle, 0, 1.8);
    // Previous motor command feedback gives the feed-forward network short-term
    // memory, making multi-step motions much easier to discover.
    for(let i=0;i<cfg.outputCount;i++) out[k++] = clamp(this.lastOut[i] || 0, -1, 1);
    return out;
  }
  applyOutputs(out, dt, sensedFeet){
    const smooth = clamp(7.5 * dt * (0.45 + this.j.muscle) * this.traits.motorSmoothing, 0, 1);
    const feet = sensedFeet || this.computeFeet();
    const lContact = feet.left.contact ? 1 : 0;
    const rContact = feet.right.contact ? 1 : 0;
    const lSwingCmd = ((out[10] || 0) + 1) * 0.5;
    const rSwingCmd = ((out[11] || 0) + 1) * 0.5;
    const lStiffCmd = ((out[12] || 0) + 1) * 0.5;
    const rStiffCmd = ((out[13] || 0) + 1) * 0.5;
    const reflex = this.traits.reflexGain;
    const lLateStance = smoothstep(0.24, 0.66, this.lStanceTime);
    const rLateStance = smoothstep(0.24, 0.66, this.rStanceTime);
    const lSwingGate = lSwingCmd * (1 - lContact * 0.58 + lLateStance * 0.32) * reflex;
    const rSwingGate = rSwingCmd * (1 - rContact * 0.58 + rLateStance * 0.32) * reflex;
    const lStanceGate = lContact * lStiffCmd * reflex;
    const rStanceGate = rContact * rStiffCmd * reflex;
    const target = {
      lh: out[0] * 0.92 + lSwingGate * 0.26 - lStanceGate * 0.10,
      lk: 0.04 + ((out[1] + 1) * 0.5) * 1.28 + lSwingGate * 0.38 - lStanceGate * 0.18,
      la: out[2] * 0.48 + lSwingGate * 0.10 - lStanceGate * 0.05,
      rh: out[3] * 0.92 + rSwingGate * 0.26 - rStanceGate * 0.10,
      rk: 0.04 + ((out[4] + 1) * 0.5) * 1.28 + rSwingGate * 0.38 - rStanceGate * 0.18,
      ra: out[5] * 0.48 + rSwingGate * 0.10 - rStanceGate * 0.05,
      ls: out[6] * 1.15,
      rs: out[7] * 1.15,
      lean: out[8] * (stage === 'run' ? 0.42 : 0.32),
      muscle: 0.35 + ((out[9] + 1) * 0.5) * 0.9 + (lStiffCmd + rStiffCmd - 1) * 0.10,
      lswing: lSwingCmd,
      rswing: rSwingCmd,
      lstiff: lStiffCmd,
      rstiff: rStiffCmd
    };
    target.lh = clamp(target.lh, -1.05, 1.05);
    target.rh = clamp(target.rh, -1.05, 1.05);
    target.lk = clamp(target.lk, 0.03, 1.45);
    target.rk = clamp(target.rk, 0.03, 1.45);
    target.la = clamp(target.la, -0.58, 0.58);
    target.ra = clamp(target.ra, -0.58, 0.58);
    target.muscle = clamp(target.muscle, 0.18, 1.35);
    for(const key of ['lh','lk','la','rh','rk','ra','ls','rs','lean','muscle','lswing','rswing','lstiff','rstiff']){
      const energyWeight = key === 'muscle' ? 0.4 : (key.endsWith('swing') || key.endsWith('stiff') ? 0.22 : 1.0);
      this.energy += Math.abs(target[key] - this.j[key]) * energyWeight;
      this.j[key] = lerp(this.j[key], target[key], smooth);
    }
    this.lastOut.set(out);
  }
  computeFeet(){
    const thigh = 0.53, shin = 0.53;
    const bodyInfluence = this.angle * 0.26;
    function leg(side, hip, knee, ankle, x, y){
      const hipA = bodyInfluence + hip;
      const kneeA = hipA - knee * 0.72 + ankle * 0.15;
      const kx = x + thigh * Math.sin(hipA);
      const ky = y - thigh * Math.cos(hipA);
      const fx = kx + shin * Math.sin(kneeA);
      const fy = ky - shin * Math.cos(kneeA);
      return {side, kneeX:kx, kneeY:ky, x:fx, y:fy, relX:fx-x, reach:Math.hypot(fx-x, fy-y), contact:fy <= 0.025 && fy > -0.18 && knee < 0.98};
    }
    return {
      left: leg('L', this.j.lh, this.j.lk, this.j.la, this.x, this.y),
      right: leg('R', this.j.rh, this.j.rk, this.j.ra, this.x, this.y)
    };
  }
  step(dt, targetSpeed){
    if(!this.alive) return;
    this.age += dt;
    const sensedFeet = this.computeFeet();
    const input = this.inputVector(targetSpeed);
    const out = this.brain.think(input);
    this.applyOutputs(out, dt, sensedFeet);

    const feet = this.computeFeet();
    const traits = this.traits;
    const contacts = [feet.left, feet.right].filter(f => f.contact);
    const contactCount = contacts.length;
    let forceX = 0;
    let forceY = -9.8;
    let support = this.x;
    let stanceMask = 0;
    const oldVx = this.vx;
    const oldVy = this.vy;

    if(feet.left.contact) stanceMask |= 1;
    if(feet.right.contact) stanceMask |= 2;
    if(contactCount === 1) this.singleContactTime += dt;
    if(contactCount === 2) this.doubleContactTime += dt;
    if(stanceMask !== 0 && stanceMask !== this.lastContactMask) this.steps++;
    this.lastContactMask = stanceMask;

    const releaseFoot = (side) => {
      const stanceKey = side === 'L' ? 'lStanceTime' : 'rStanceTime';
      const plantKey = side === 'L' ? 'lPlantX' : 'rPlantX';
      const maxTravelKey = side === 'L' ? 'lMaxTravel' : 'rMaxTravel';
      if(this[plantKey] !== null && this[stanceKey] > 0){
        const goodStance = this[stanceKey] > 0.12 && this[stanceKey] < 1.15 && this[maxTravelKey] > 0.075;
        if(goodStance){
          this.goodSteps++;
          if(side === 'L') this.leftGoodSteps++; else this.rightGoodSteps++;
          if(this.lastGoodFoot && this.lastGoodFoot !== side) this.altGoodSteps++;
          if(this.lastGoodFoot === side) this.sameFootGoodSteps++;
          this.lastGoodFoot = side;
          this.bestDualStepMin = Math.min(this.leftGoodSteps, this.rightGoodSteps);
        }
      }
      this[plantKey] = null; this[stanceKey] = 0; this[maxTravelKey] = 0;
    };

    if(contactCount){
      support = contacts.reduce((a,f)=>a+f.x,0) / contactCount;
      this.groundTime += dt;
      if(contactCount === 2){
        const bothFeetSpan = Math.abs(feet.left.x - feet.right.x);
        const bothFeetStill = Math.max(Math.abs(feet.left.x - this.prevLX), Math.abs(feet.right.x - this.prevRX)) / dt;
        // Keeping both feet glued down can look stable, but it is not a useful
        // stepping strategy. Penalize long double-support while moving or while
        // the feet are almost motionless, so actual alternating support can win.
        if(this.age > 0.8){
          const excessDouble = Math.max(0, this.doubleContactTime / Math.max(0.1, this.age) - (stage === 'run' ? 0.30 : 0.58));
          const stuckWhileMoving = Math.max(0, Math.abs(this.vx) - 0.18) * (1 - smoothstep(0.06, 0.22, bothFeetStill));
          const narrowBase = Math.max(0, 0.16 - bothFeetSpan);
          this.gluedFeetPenalty += (excessDouble * excessDouble * 11.0 + stuckWhileMoving * 1.6 + narrowBase * 0.7) * dt;
        }
      }
      for(const foot of contacts){
        const isLeft = foot.side === 'L';
        const prevRel = isLeft ? this.prevRelL : this.prevRelR;
        const prevX = isLeft ? this.prevLX : this.prevRX;
        const relVel = (foot.relX - prevRel) / dt;
        const footVelX = (foot.x - prevX) / dt;
        const plantKey = isLeft ? 'lPlantX' : 'rPlantX';
        const stanceKey = isLeft ? 'lStanceTime' : 'rStanceTime';
        const touchRelKey = isLeft ? 'lTouchRel' : 'rTouchRel';
        const maxTravelKey = isLeft ? 'lMaxTravel' : 'rMaxTravel';
        if(this[plantKey] === null || this[stanceKey] <= 0){
          this[plantKey] = foot.x;
          this[touchRelKey] = foot.relX;
          this[maxTravelKey] = 0;
          if(isLeft) this.leftTouchCount++; else this.rightTouchCount++;
        }
        this[stanceKey] += dt;
        if(isLeft) this.leftContactTime += dt; else this.rightContactTime += dt;
        this[maxTravelKey] = Math.max(this[maxTravelKey], Math.abs(foot.relX - this[touchRelKey]));

        const penetration = clamp(0.025 - foot.y, 0, 0.12);
        const sideStiff = isLeft ? this.j.lstiff : this.j.rstiff;
        const sideSwing = isLeft ? this.j.lswing : this.j.rswing;
        const rawSpring = (120 + sideStiff * 32) * penetration - (6.8 + sideStiff * 1.2) * this.vy;
        const spring = Math.max(0, rawSpring);
        forceY += spring / contactCount;
        if(foot.y < -0.08) this.hoverPenalty += (-0.08 - foot.y) * (-0.08 - foot.y) * 18 * dt;

        const prevRelVel = isLeft ? this.prevRelVelL : this.prevRelVelR;
        const relAccel = (relVel - prevRelVel) / dt;
        const sweepExcess = Math.max(0, Math.abs(relVel) - 3.8);
        const relAccelExcess = Math.max(0, Math.abs(relAccel) - 72);
        this.jitterPenalty += (sweepExcess * sweepExcess * 0.060 + relAccelExcess * relAccelExcess * 0.00055) * dt;

        const plantSlip = foot.x - this[plantKey];
        const backwardFootSpeed = clamp(-footVelX, 0, 3.4);
        const loadGate = smoothstep(1.2, 6.5, spring);
        const stanceGate = smoothstep(0.10, 0.24, this[stanceKey]);
        const plantedGate = 1 - smoothstep(0.055, 0.22, Math.abs(plantSlip));
        const travelGate = smoothstep(0.055, 0.18, this[maxTravelKey]);
        const heightGate = smoothstep(0.74, 0.92, this.y) * (1 - smoothstep(1.16, 1.30, this.y));
        const reachGate = 1 - smoothstep(1.00, 1.14, foot.reach);
        const swingPenaltyGate = 1 - sideSwing * 0.22;
        const driveGate = smoothstep(0.18, 0.70, backwardFootSpeed) * loadGate * stanceGate * plantedGate * travelGate * heightGate * reachGate * swingPenaltyGate;
        this.loadedContactTime += loadGate * stanceGate * dt / contactCount;
        this.driveContactTime += driveGate * dt / contactCount;

        // Horizontal drive now needs: a loaded stance foot, some stance duration,
        // real backward sweep, small plant slip, and a reachable body height. Tiny
        // high-frequency toe flicks fail those gates and become penalties instead.
        const rawSweep = -relVel * traits.sweepGain - this.vx * traits.speedDamp + traits.sweepBias;
        const backwardSweep = clamp(rawSweep, 0, 2.45) * driveGate;
        const behindDrive = clamp(-foot.relX * 1.15, 0, 0.92) * driveGate;
        const extension = clamp(1.02 - (isLeft ? this.j.lk : this.j.rk), 0, 0.95) * driveGate;
        const anklePop = Math.max(0, isLeft ? -this.j.la : -this.j.ra) * driveGate;

        // Do not let a one-legged specialist keep farming distance. Once one
        // side has touched or carried the body over 2x more than the other,
        // that overused side loses thrust and racks up a large penalty until
        // the neglected leg starts contributing.
        const sideTouches = isLeft ? this.leftTouchCount : this.rightTouchCount;
        const otherTouches = isLeft ? this.rightTouchCount : this.leftTouchCount;
        const sideTime = isLeft ? this.leftContactTime : this.rightContactTime;
        const otherTime = isLeft ? this.rightContactTime : this.leftContactTime;
        const touchOveruse = Math.max(0, sideTouches - Math.max(1, otherTouches) * 2);
        const timeOveruse = Math.max(0, sideTime - Math.max(0.18, otherTime) * 2.0);
        const dominanceGate = 1 / (1 + touchOveruse * 2.5 + timeOveruse * 3.5);
        const stanceStiffnessBoost = 0.86 + sideStiff * 0.28;
        const desiredDrive = (backwardSweep * 1.05 * traits.sweepForce + behindDrive * traits.behindDrive + extension * traits.extensionDrive + anklePop * traits.ankleDrive) * this.j.muscle * dominanceGate * stanceStiffnessBoost / contactCount;
        forceX += desiredDrive;
        if(touchOveruse > 0 || timeOveruse > 0){
          const harshness = stage === 'run' ? 8.5 : 5.2;
          this.legDominancePenalty += (touchOveruse * touchOveruse * 0.35 + timeOveruse * timeOveruse * 2.2) * harshness * dt;
        }
        const badDriveGate = smoothstep(0.28, 0.70, backwardFootSpeed) * (1 - driveGate);
        this.badThrustPenalty += badDriveGate * (Math.abs(rawSweep) + Math.abs(relAccel) * 0.010 + Math.max(0, this.vx)) * 0.055 * dt;

        const forwardSkate = Math.max(0, footVelX - 0.05);
        const lockedFootGlide = Math.max(0, Math.abs(this.vx) - backwardFootSpeed - 0.10);
        this.skatePenalty += (forwardSkate * forwardSkate * 0.26 + lockedFootGlide * lockedFootGlide * (0.08 + 0.18 * (1 - driveGate)) + Math.abs(plantSlip) * 0.055) * dt;

        // Strong but capped stance friction. It can carry the body over a planted
        // foot, but not act like an unlimited invisible motor.
        const anchorGate = loadGate * stanceGate * heightGate * reachGate * plantedGate;
        forceX += clamp(-plantSlip * 18 - footVelX * 1.55, -2.25, 2.25) * anchorGate / contactCount;
        if(isLeft) this.prevRelVelL = relVel; else this.prevRelVelR = relVel;
      }
      if(!feet.left.contact) releaseFoot('L');
      if(!feet.right.contact) releaseFoot('R');
      this.airStreak = 0;
      forceX -= this.vx * 0.48;
    } else {
      this.airTime += dt;
      this.airStreak += dt;
      releaseFoot('L'); releaseFoot('R');
      const fastAir = Math.max(0, Math.abs(this.vx) - 0.28);
      this.hoverPenalty += (fastAir * fastAir * 0.70 + Math.max(0, this.airStreak - 0.22) * Math.max(0, this.vx) * 1.6) * dt;
      forceX -= this.vx * (stage === 'run' ? 0.24 : 0.44);
      forceX -= this.vx * Math.abs(this.vx) * (stage === 'run' ? 0.08 : 0.14);
    }

    // Quadratic air resistance: tiny at walking speed, noticeable when evolved gaits start exploiting high speed jitter.
    forceX -= this.vx * Math.abs(this.vx) * traits.airDragX;
    forceY -= this.vy * Math.abs(this.vy) * traits.airDragX * 0.45;

    if(stage === 'walk' && contactCount === 0 && this.age > 0.6) forceY -= 2.2;

    this.vx += forceX * dt;
    this.vy += forceY * dt;
    const horizontalAccel = Math.abs((this.vx - oldVx) / dt);
    const accelExcess = Math.max(0, horizontalAccel - traits.accelTolerance);
    this.accelPenalty += accelExcess * accelExcess * dt * 0.040;
    this.vx = clamp(this.vx, -1.3, stage === 'run' ? 6.0 : 3.4);
    this.y += this.vy * dt;
    if(this.y > 1.34){ this.y = 1.34; this.vy *= 0.32; }

    if(contactCount && this.y < 0.86){
      this.y = lerp(this.y, 0.94, 0.08);
      this.vy = Math.max(this.vy, 0);
    }
    if(contactCount && this.y > 1.22){
      this.hoverPenalty += (this.y - 1.18) * (this.y - 1.18) * 46 * dt;
    }
    this.x += this.vx * dt;
    this.maxX = Math.max(this.maxX, this.x);
    this.minY = Math.min(this.minY, this.y);

    const balance = clamp(this.x - support, -1.25, 1.25);
    const leanTarget = this.j.lean + clamp((targetSpeed - this.vx) * 0.035, -0.12, 0.12);
    const activePosture = 0.85 + this.j.muscle * 1.35;
    const balanceTorque = contactCount ? balance * (1.25 + this.j.muscle * 0.85) : 0.06 * Math.sign(this.vx || 1);
    this.av += ((leanTarget - this.angle) * activePosture + balanceTorque - this.angle * 0.38 - this.av * 0.16) * dt;
    this.angle += this.av * dt;

    // Natural disasters for impossible gaits.
    if(this.age > 0.7 && (Math.abs(this.angle) > 1.15 || this.y < 0.52 || this.x < -1.2 || Math.abs(balance) > 1.08 && contactCount || this.airStreak > (stage === 'run' ? 0.72 : 0.42) && Math.abs(this.vx) > 0.55)){
      this.alive = false;
      this.fallen = true;
    }
    if(!Number.isFinite(this.x + this.y + this.vx + this.angle)){
      this.alive = false; this.fallen = true; this.x = -10;
    }
    this.avgSpeed = this.age > 0 ? this.x / this.age : 0;
    this.fitness = this.score(targetSpeed, false);

    this.prevRelL = feet.left.relX; this.prevRelR = feet.right.relX;
    if(!feet.left.contact) this.prevRelVelL *= 0.82;
    if(!feet.right.contact) this.prevRelVelR *= 0.82;
    this.prevLX = feet.left.x; this.prevRX = feet.right.x;
    this.prevVx = this.vx; this.prevVy = this.vy;
    if(this.trail.length === 0 || Math.abs(this.trail[this.trail.length-1].x - this.x) > 0.18){
      this.trail.push({x:this.x, y:this.y});
      if(this.trail.length > 80) this.trail.shift();
    }
  }
  score(targetSpeed, final){
    const timeLimit = stage === 'walk' ? cfg.walkTime : cfg.runTime;
    const survival = clamp(this.age / timeLimit, 0, 1);
    const dist = Math.max(0, this.x);
    const avg = this.age > 0.1 ? this.x / this.age : 0;
    const speedScore = Math.max(0, 1 - Math.abs(avg - targetSpeed) / Math.max(targetSpeed, 0.1));
    const upright = Math.max(0, 1 - Math.abs(this.angle) / 1.05);
    const leftSteps = this.leftGoodSteps || 0;
    const rightSteps = this.rightGoodSteps || 0;
    const altSteps = this.altGoodSteps || 0;
    const maxSideSteps = Math.max(leftSteps, rightSteps);
    const minSideSteps = Math.min(leftSteps, rightSteps);
    const legBalance = maxSideSteps > 0 ? minSideSteps / maxSideSteps : 0;
    const leftTouches = this.leftTouchCount || 0;
    const rightTouches = this.rightTouchCount || 0;
    const maxTouches = Math.max(leftTouches, rightTouches);
    const minTouches = Math.min(leftTouches, rightTouches);
    const touchBalance = maxTouches > 0 ? minTouches / maxTouches : 0;
    const maxContactT = Math.max(this.leftContactTime || 0, this.rightContactTime || 0);
    const minContactT = Math.min(this.leftContactTime || 0, this.rightContactTime || 0);
    const contactBalance = maxContactT > 0 ? minContactT / maxContactT : 0;
    const dominanceTouchExcess = Math.max(0, maxTouches - Math.max(1, minTouches) * 2);
    const dominanceTimeExcess = Math.max(0, maxContactT - Math.max(0.18, minContactT) * 2.0);
    const dominancePenalty = (dominanceTouchExcess * dominanceTouchExcess * (stage === 'run' ? 13.0 : 8.0) + dominanceTimeExcess * dominanceTimeExcess * (stage === 'run' ? 46.0 : 28.0)) * clamp(dist / 3.0, 0.25, 1.8);
    const generalStepScore = clamp((this.goodSteps * 0.70 + this.steps * 0.10) / (stage === 'run' ? 18 : 13), 0, 1);
    const bipedStepScore = clamp((minSideSteps * 1.15 + altSteps * 0.85) / (stage === 'run' ? 12 : 8), 0, 1);
    // Quality should come mostly from real left/right step releases, not from
    // simply touching both feet to the floor. Touch/contact balance is a safety
    // gate, but it is no longer a big positive reward by itself.
    const bipedQuality = clamp(legBalance * 0.46 + bipedStepScore * 0.54, 0, 1);
    const twoLegUseScore = clamp((minSideSteps * 1.25 + altSteps * 1.05 + Math.min(minTouches, 6) * 0.18) / (stage === 'run' ? 13.0 : 8.6), 0, 1);
    const softMonoCap = stage === 'run' ? 5.0 : 3.2;
    const progressGate = clamp(0.28 + bipedQuality * 0.72, 0.28, 1);
    const bipedDist = Math.min(dist, softMonoCap) + Math.max(0, dist - softMonoCap) * progressGate;
    const sameFootPenalty = (this.sameFootGoodSteps || 0) * (stage === 'run' ? 1.0 : 1.35);
    const oneLegDistancePenalty = Math.max(0, dist - softMonoCap) * (1 - legBalance) * (stage === 'run' ? 6.5 : 8.0);
    const energyPenalty = this.energy * (stage === 'run' ? 0.015 : 0.019);
    const contactHealth = clamp((this.loadedContactTime + this.driveContactTime * 1.8) / Math.max(0.1, this.age), 0, 1);
    const unsupportedSpeedPenalty = Math.max(0, avg - 0.45) * Math.max(0, avg - 0.45) * (1 - contactHealth) * (stage === 'run' ? 10 : 18);
    const hoverRatio = this.airTime / Math.max(0.1, this.age);
    const doubleRatio = this.doubleContactTime / Math.max(0.1, this.age);
    const singleRatio = this.singleContactTime / Math.max(0.1, this.age);
    const maxDouble = stage === 'run' ? 0.34 : 0.62;
    const doubleSupportPenalty = Math.max(0, doubleRatio - maxDouble) * Math.max(0, doubleRatio - maxDouble) * (stage === 'run' ? 34 : 22) * (0.5 + survival);
    const contactMixScore = stage === 'run'
      ? clamp(singleRatio * 1.3 + hoverRatio * 0.5 - Math.max(0, doubleRatio - 0.22) * 1.6, 0, 1)
      : clamp(singleRatio * 0.9 + Math.min(doubleRatio, 0.45) * 0.35 - Math.max(0, doubleRatio - 0.62) * 1.2, 0, 1);
    const antiExploitPenalty = this.accelPenalty + this.jitterPenalty + this.skatePenalty + this.hoverPenalty + this.badThrustPenalty + this.legDominancePenalty + this.gluedFeetPenalty + unsupportedSpeedPenalty + oneLegDistancePenalty + dominancePenalty + sameFootPenalty + doubleSupportPenalty;
    const fallPenalty = this.fallen ? (stage === 'run' ? 9 : 7) : 0;
    let fit = 0;
    if(stage === 'walk'){
      const walkContacts = this.groundTime / Math.max(0.1, this.age);
      fit = bipedDist * 11.4 + survival * 7 + speedScore * 7 + upright * 4 + generalStepScore * 1.4 + bipedStepScore * 15 + twoLegUseScore * 7 + legBalance * 3 + contactMixScore * 5 + contactHealth * 3.4 - hoverRatio * 14 - energyPenalty - antiExploitPenalty - fallPenalty;
      // Let imperfect movers survive long enough to become alternating walkers.
      // The big bonus still needs real biped stepping, but raw progress matters.
      if(dist > 1.0 && avg > 0.18) fit += Math.min(dist, cfg.trackWinWalk) * 1.7;
      if(avg > 0.85 && dist > cfg.trackWinWalk && leftSteps >= 3 && rightSteps >= 3 && altSteps >= 4) fit += 35;
      if(dist > 7 && (leftSteps < 2 || rightSteps < 2)) fit -= (dist - 7) * 13;
      if(dist > 3 && maxTouches > 3 && touchBalance < 0.44) fit -= (dist - 3) * (0.44 - touchBalance) * 23;
      if(avg > 2.05) fit -= (avg - 2.05) * 10;
    } else {
      const airRatio = this.airTime / Math.max(0.1, this.age);
      fit = bipedDist * 13.2 + survival * 6 + speedScore * 12 + upright * 3.5 + generalStepScore * 1.4 + bipedStepScore * 13 + twoLegUseScore * 8 + legBalance * 2.5 + contactMixScore * 4 + contactHealth * 3.2 - Math.max(0, airRatio - 0.46) * 18 - energyPenalty - antiExploitPenalty - fallPenalty;
      if(dist > 1.5 && avg > 0.35) fit += Math.min(dist, cfg.trackWinRun) * 1.3;
      if(avg > 2.45 && dist > cfg.trackWinRun && leftSteps >= 4 && rightSteps >= 4 && altSteps >= 6 && touchBalance >= 0.50 && contactBalance >= 0.42) fit += 70;
      if(dist > 9 && (leftSteps < 3 || rightSteps < 3)) fit -= (dist - 9) * 13;
      if(dist > 4 && maxTouches > 3 && touchBalance < 0.46) fit -= (dist - 4) * (0.46 - touchBalance) * 46;
      if(dist > 4 && contactBalance < 0.38) fit -= (dist - 4) * (0.38 - contactBalance) * 36;
    }
    if(this.x < 0) fit += this.x * 16;
    if(final && !this.fallen) fit += 5;
    return fit;
  }
}
