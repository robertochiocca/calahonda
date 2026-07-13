window.addEventListener('scroll', () => {
  const d = document.documentElement;
  document.getElementById('pb').style.width = (d.scrollTop / (d.scrollHeight - d.clientHeight) * 100) + '%';
}, { passive: true });

(function () {
  const rows = document.querySelectorAll('.sc-row[data-weight]');
  let animated = false;
  function animateScorecard() {
    if (animated) return;
    animated = true;
    let total = 0;
    rows.forEach(row => {
      const w = +row.dataset.weight;
      const s = +row.dataset.score;
      total += w * s / 100;
      const bar = row.querySelector('.sc-bar-fill');
      setTimeout(() => { bar.style.width = (s * 10) + '%'; }, 80);
    });
    setTimeout(() => {
      document.getElementById('sc-total').textContent = total.toFixed(1) + '/10';
    }, 500);
  }
  const sc = document.getElementById('scorecard');
  if (sc) {
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) animateScorecard();
    }, { threshold: 0.3 });
    observer.observe(sc);
  }
})();

function setLang(lang) {
  document.body.classList.toggle('en', lang === 'en');
  document.documentElement.lang = lang === 'en' ? 'en' : 'pt-BR';
  document.getElementById('btn-pt').classList.toggle('active', lang === 'pt');
  document.getElementById('btn-en').classList.toggle('active', lang === 'en');
  try { localStorage.setItem('calahonda-lang', lang); } catch (e) { /* modo privado */ }
}

// Restaura o idioma escolhido na última visita
try {
  const saved = localStorage.getItem('calahonda-lang');
  if (saved === 'en') setLang('en');
} catch (e) { /* modo privado */ }


/* ===== Demo interativa de VaR =====
   Mesma matemática do módulo Python (calahonda_var), com parâmetros
   diários ilustrativos — os do gerador sintético — e correlação 0.35. */

const DEMO_ASSETS = {
  PETR4: { mu: -0.0004, vol: 0.022 },
  VALE3: { mu: -0.0002, vol: 0.020 },
  ITUB4: { mu:  0.0002, vol: 0.016 },
  BBAS3: { mu:  0.0001, vol: 0.017 },
  WEGE3: { mu:  0.0004, vol: 0.018 },
  B3SA3: { mu:  0.0001, vol: 0.021 }
};
const DEMO_CORR = 0.35;
const DEMO_Z = { '0.90': 1.2816, '0.95': 1.6449, '0.99': 2.3263 };
const DEMO_N_SIMS = 10000;

const isEN = () => document.body.classList.contains('en');
const fmtBRL = v => v.toLocaleString('pt-BR', { style: 'currency', currency: 'BRL', maximumFractionDigits: 0 });
const fmtPct = v => (v * 100).toFixed(2).replace('.', isEN() ? '.' : ',') + '%';

// PRNG com semente (mulberry32) + Box-Muller: simulação reprodutível
function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gaussian(rand) {
  const u = 1 - rand(), v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
const normPdf = z => Math.exp(-z * z / 2) / Math.sqrt(2 * Math.PI);

let demoState = null;

function runDemo() {
  // 1. Carteira: pesos normalizados
  const rows = document.querySelectorAll('#demo .demo-row');
  const assets = [], weights = [];
  rows.forEach(row => {
    const sel = row.querySelector('select');
    const inp = row.querySelector('input');
    if (!sel || !inp) return;
    const w = parseFloat(inp.value) || 0;
    if (w > 0) { assets.push(sel.value); weights.push(w); }
  });
  const total = weights.reduce((a, b) => a + b, 0);
  if (!total) return;
  const w = weights.map(x => x / total);

  const value = Math.max(1000, parseFloat(document.getElementById('demo-value').value) || 1e6);
  const confStr = document.getElementById('demo-conf').value;
  const conf = parseFloat(confStr);
  const h = Math.min(252, Math.max(1, parseInt(document.getElementById('demo-horizon').value) || 21));

  // 2. Parâmetros da carteira: μ = wᵀμ ; σ² = wᵀΣw (Σ via correlação constante)
  let mu = 0;
  for (let i = 0; i < assets.length; i++) mu += w[i] * DEMO_ASSETS[assets[i]].mu;
  let varDaily = 0;
  for (let i = 0; i < assets.length; i++) {
    for (let j = 0; j < assets.length; j++) {
      const rho = i === j ? 1 : DEMO_CORR;
      varDaily += w[i] * w[j] * rho * DEMO_ASSETS[assets[i]].vol * DEMO_ASSETS[assets[j]].vol;
    }
  }
  const sigma = Math.sqrt(varDaily);
  const muH = mu * h, sigH = sigma * Math.sqrt(h);

  // 3. Paramétrico (fórmula fechada, normal)
  const z = DEMO_Z[confStr], alpha = 1 - conf;
  const varPar = z * sigH - muH;
  const cvarPar = sigH * normPdf(z) / alpha - muH;

  // 4. Monte Carlo: 10.000 cenários de P&L no horizonte
  const rand = mulberry32(42);
  const pnl = new Float64Array(DEMO_N_SIMS);
  for (let i = 0; i < DEMO_N_SIMS; i++) pnl[i] = muH + sigH * gaussian(rand);
  const sorted = Float64Array.from(pnl).sort();
  const varMC = -sorted[Math.floor(alpha * DEMO_N_SIMS)];
  let tailSum = 0, tailN = 0;
  for (let i = 0; i < DEMO_N_SIMS && sorted[i] <= -varMC; i++) { tailSum += sorted[i]; tailN++; }
  const cvarMC = tailN ? -tailSum / tailN : varMC;

  // 5. Render
  document.getElementById('demo-empty').style.display = 'none';
  document.getElementById('demo-out').style.display = 'block';
  document.getElementById('demo-var-brl').textContent = fmtBRL(varMC * value);
  document.getElementById('demo-var-pct').textContent =
    fmtPct(varMC) + (isEN() ? ` of the portfolio · ${h} days · ${Math.round(conf * 100)}%` : ` da carteira · ${h} dias · ${Math.round(conf * 100)}%`);
  document.getElementById('demo-cvar-brl').textContent = fmtBRL(cvarMC * value);
  document.getElementById('demo-cvar-pct').textContent =
    fmtPct(cvarMC) + (isEN() ? ' — mean loss beyond the VaR' : ' — perda média além do VaR');

  const rowsHtml = [
    ['parametric', varPar, cvarPar],
    ['monte_carlo', varMC, cvarMC]
  ].map(([name, v, c]) =>
    `<tr><td>${name}</td><td>${fmtPct(v)}</td><td>${fmtPct(c)}</td><td>${fmtBRL(v * value)}</td></tr>`
  ).join('');
  document.getElementById('demo-tbody').innerHTML = rowsHtml;

  document.getElementById('demo-chart-note').textContent = isEN()
    ? `Distribution of simulated P&L over ${h} trading days (${DEMO_N_SIMS.toLocaleString('en-US')} scenarios). Amber line = VaR · red line = CVaR.`
    : `Distribuição do P&L simulado em ${h} dias úteis (${DEMO_N_SIMS.toLocaleString('pt-BR')} cenários). Linha âmbar = VaR · vermelha = CVaR.`;

  demoState = { pnl: sorted, varMC, cvarMC, h };
  drawDemoHistogram();
}

function drawDemoHistogram() {
  if (!demoState) return;
  const { pnl, varMC, cvarMC } = demoState;
  const canvas = document.getElementById('demo-canvas');
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth, H = canvas.clientHeight;
  canvas.width = W * dpr; canvas.height = H * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);

  const padL = 8, padR = 8, padT = 26, padB = 22;
  const min = pnl[0], max = pnl[pnl.length - 1];
  const nBins = 48, binW = (max - min) / nBins;
  const bins = new Array(nBins).fill(0);
  pnl.forEach(v => bins[Math.min(nBins - 1, Math.floor((v - min) / binW))]++);
  const maxBin = Math.max(...bins);
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const xOf = v => padL + ((v - min) / (max - min)) * plotW;

  // grid recessivo
  ctx.strokeStyle = 'rgba(255,255,255,.06)';
  ctx.lineWidth = 1;
  [0.25, 0.5, 0.75].forEach(f => {
    ctx.beginPath();
    ctx.moveTo(padL, padT + plotH * f); ctx.lineTo(W - padR, padT + plotH * f);
    ctx.stroke();
  });

  // barras (série única, lima) com respiro de 2px entre elas
  const barW = plotW / nBins;
  for (let i = 0; i < nBins; i++) {
    const bh = (bins[i] / maxBin) * plotH;
    ctx.fillStyle = '#C8F135';
    ctx.globalAlpha = 0.88;
    ctx.fillRect(padL + i * barW + 1, padT + plotH - bh, Math.max(1, barW - 2), bh);
  }
  ctx.globalAlpha = 1;

  // linha de base + rótulos min/0/max
  ctx.strokeStyle = 'rgba(255,255,255,.18)';
  ctx.beginPath(); ctx.moveTo(padL, padT + plotH + .5); ctx.lineTo(W - padR, padT + plotH + .5); ctx.stroke();
  ctx.fillStyle = '#8892A4';
  ctx.font = '10px "IBM Plex Mono", monospace';
  ctx.textAlign = 'left';
  ctx.fillText((min * 100).toFixed(0) + '%', padL, H - 7);
  ctx.textAlign = 'right';
  ctx.fillText('+' + (max * 100).toFixed(0) + '%', W - padR, H - 7);
  if (min < 0 && max > 0) { ctx.textAlign = 'center'; ctx.fillText('0', xOf(0), H - 7); }

  // limiares com rótulo direto (nunca cor sozinha)
  const threshold = (v, color, label, line) => {
    const x = xOf(-v);
    ctx.strokeStyle = color; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(x, padT - 4); ctx.lineTo(x, padT + plotH); ctx.stroke();
    ctx.fillStyle = color; ctx.textAlign = x < W / 2 ? 'left' : 'right';
    ctx.fillText(label, x + (x < W / 2 ? 5 : -5), 12 + line * 12);
  };
  const confLabel = document.getElementById('demo-conf').selectedOptions[0].textContent;
  threshold(varMC, '#F5A623', `VaR ${confLabel} = ${(varMC * 100).toFixed(1)}%`, 0);
  threshold(cvarMC, '#F24B4B', `CVaR = ${(cvarMC * 100).toFixed(1)}%`, 1);

  // camada de hover: tooltip por barra
  canvas.onmousemove = e => {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const i = Math.floor((x - padL) / barW);
    const tt = document.getElementById('demo-tt');
    if (i < 0 || i >= nBins) { tt.style.display = 'none'; return; }
    const lo = (min + i * binW) * 100, hi = (min + (i + 1) * binW) * 100;
    tt.textContent = isEN()
      ? `${lo.toFixed(1)}% to ${hi.toFixed(1)}% · ${bins[i]} scenarios`
      : `${lo.toFixed(1)}% a ${hi.toFixed(1)}% · ${bins[i]} cenários`;
    tt.style.display = 'block';
    const wrap = canvas.parentElement.getBoundingClientRect();
    const left = Math.min(e.clientX - wrap.left + 12, wrap.width - tt.offsetWidth - 8);
    tt.style.left = left + 'px';
    tt.style.top = (e.clientY - wrap.top - 34) + 'px';
  };
  canvas.onmouseleave = () => { document.getElementById('demo-tt').style.display = 'none'; };
}

window.addEventListener('resize', () => { if (demoState) drawDemoHistogram(); });

/* Revela as seções ao entrarem na viewport (acessível: respeita
   prefers-reduced-motion) */
if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
  const revealObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        revealObserver.unobserve(entry.target);
      }
    });
  }, { threshold: 0.08 });
  document.querySelectorAll('.si, .cta-inner').forEach(el => {
    el.classList.add('reveal');
    revealObserver.observe(el);
  });
}
