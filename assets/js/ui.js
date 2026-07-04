// DOM HUD updates and control state display.
'use strict';

function updateUI(){
  const alive = population.filter(a=>a.alive).length;
  const bestDistance = bestEver ? Math.max(0,bestEver.x) : 0;
  const bestSpeed = bestEver ? Math.max(0,bestEver.avgSpeed) : 0;
  els.generation.textContent = generation;
  els.alive.textContent = `${alive}/${cfg.popSize}`;
  els.bestDist.textContent = m(bestDistance);
  els.bestSpeed.textContent = ms(bestSpeed);
  els.diversity.textContent = stagnation.lastDiversity.toFixed(2);
  els.escapes.textContent = stagnation.escapes;
  els.escapeStatus.textContent = stagnation.lastAction === 'normal' ? `stale ${stagnation.stale}` : `${stagnation.lastAction} · stale ${stagnation.stale}`;
  els.episodeClock.textContent = `${episodeTime.toFixed(1)} s / ${timeLimit().toFixed(1)} s`;
  els.stageBadge.textContent = won ? 'Stage: run mastered' : `Stage: ${stage}`;
  els.stageBadge.style.borderColor = stage === 'run' ? 'rgba(167,243,208,.45)' : 'rgba(125,211,252,.35)';
  const bestGait = bestEver ? gaitStats(bestEver) : {left:0,right:0,alt:0,balance:0};
  const walkGaitProgress = Math.min(1, Math.min(bestGait.left / 3, bestGait.right / 3, bestGait.alt / 4, bestGait.balance));
  const runGaitProgress = Math.min(1, Math.min(bestGait.left / 4, bestGait.right / 4, bestGait.alt / 6, bestGait.balance));
  const walkProgress = Math.min(1, Math.min(bestDistance / cfg.trackWinWalk, bestSpeed / 0.85, walkGaitProgress));
  const runProgress = Math.min(1, Math.min(bestDistance / cfg.trackWinRun, bestSpeed / 2.45, runGaitProgress));
  els.walkBar.style.width = `${Math.round(walkProgress*100)}%`;
  els.runBar.style.width = `${Math.round(runProgress*100)}%`;
  els.walkPct.textContent = `${Math.round(walkProgress*100)}%`;
  els.runPct.textContent = `${Math.round(runProgress*100)}%`;
  els.speedVal.textContent = `${els.speed.value}×`;
  els.mutVal.textContent = `${Number(els.mutation.value).toFixed(1)}%`;
  els.toggle.textContent = paused ? 'Resume' : 'Pause';
}
