'use strict';

window.GANN_DOM = {
  canvas: document.getElementById('world'),
  chart: document.getElementById('chart'),
  els: {
    toggle: document.getElementById('toggle'),
    reset: document.getElementById('reset'),
    cold: document.getElementById('cold'),
    forceRun: document.getElementById('forceRun'),
    speed: document.getElementById('speed'),
    speedVal: document.getElementById('speedVal'),
    mutation: document.getElementById('mutation'),
    mutVal: document.getElementById('mutVal'),
    cameraMode: document.getElementById('cameraMode'),
    generation: document.getElementById('generation'),
    alive: document.getElementById('alive'),
    bestDist: document.getElementById('bestDist'),
    bestSpeed: document.getElementById('bestSpeed'),
    diversity: document.getElementById('diversity'),
    escapes: document.getElementById('escapes'),
    escapeStatus: document.getElementById('escapeStatus'),
    walkPct: document.getElementById('walkPct'),
    runPct: document.getElementById('runPct'),
    walkBar: document.getElementById('walkBar'),
    runBar: document.getElementById('runBar'),
    episodeClock: document.getElementById('episodeClock'),
    stageBadge: document.getElementById('stageBadge'),
    save: document.getElementById('save'),
    load: document.getElementById('load')
  }
};
