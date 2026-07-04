'use strict';

function clamp(v, lo, hi){ return Math.max(lo, Math.min(hi, v)); }
function lerp(a, b, t){ return a + (b - a) * t; }
function smoothstep(edge0, edge1, x){
  const t = clamp((x - edge0) / Math.max(1e-9, edge1 - edge0), 0, 1);
  return t * t * (3 - 2 * t);
}
const tanh = Math.tanh ? Math.tanh : function tanhFallback(x){ return (Math.exp(2 * x) - 1) / (Math.exp(2 * x) + 1); };
function m(v){ return `${v.toFixed(1)} m`; }
function ms(v){ return `${v.toFixed(2)} m/s`; }

window.GANN_MATH = {clamp, lerp, smoothstep, tanh, m, ms};
