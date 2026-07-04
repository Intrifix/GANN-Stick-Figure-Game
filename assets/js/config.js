'use strict';

window.GANN_CONFIG = Object.freeze({
  popSize: 42,
  eliteCount: 5,
  inputCount: 49,
  hiddenCount: 30,
  outputCount: 14,
  traitCount: 13,
  dt: 1 / 60,
  walkTime: 18,
  runTime: 13.5,
  walkTarget: 1.25,
  runTarget: 2.8,
  trackWinWalk: 18,
  trackWinRun: 34,
  scale: 72,
  storageKeys: Object.freeze([
    'gann-stick-learner-champion-v4',
    'gann-stick-learner-champion-v3',
    'gann-stick-learner-champion-v2',
    'gann-stick-learner-champion-v1'
  ])
});
