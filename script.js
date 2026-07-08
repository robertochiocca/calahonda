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

const posts = {
  1: `Como estudante querendo entrar em finanças quantitativas, decidi parar de só estudar teoria e construir algo real.\n\nO resultado: um módulo de Value at Risk (VaR) em Python, do zero.\n\nEle calcula o risco de uma carteira de ações da B3 por três métodos:\n📊 Histórico (distribuição empírica)\n📈 Paramétrico (variância-covariância)\n🎲 Monte Carlo (10.000 simulações)\n\nMais o CVaR (Expected Shortfall) e 8 testes unitários — incluindo um que valida o resultado contra a teoria da distribuição normal.\n\nStack: Python, NumPy, pandas, SciPy.\n\nFaz parte de um projeto maior que estou usando para estudar o mercado brasileiro: a Calahonda — um conceito de plataforma quant focada na B3. Por enquanto é portfólio, não produto, e sou honesto sobre isso: o VaR funciona, o resto é roadmap.\n\nCódigo aberto no GitHub (link na bio). Feedback de quem trabalha com risco ou quant é muito bem-vindo.\n\nQual métrica de risco vocês consideram mais útil no dia a dia?\n\n#QuantFinance #Python #DataScience #RiskManagement #B3 #VaR`,
  2: `Value at Risk (VaR) é a métrica de risco mais usada por fundos no mundo.\n\nEla responde: "qual é a perda máxima esperada com 95% de confiança nos próximos 21 dias?"\n\nA maioria dos fundos menores no Brasil ainda calcula isso manualmente. Resolvi automatizar.\n\nO método (Monte Carlo, 4 passos):\n1. Dados históricos da B3 via yfinance\n2. Retornos diários de cada ativo\n3. 10.000 simulações de Monte Carlo\n4. Percentil 5% da distribuição de perdas\n\nResultado (PETR4 + VALE3 + ITUB4):\n→ VaR 95% (21 dias) = R$ 103.656\n→ CVaR = R$ 130.743\n→ Tempo: 1.2 segundos\n\nPara uma gestora que fazia isso em Excel: 3–4 horas economizadas por semana.\n\nEsse é o Módulo 03 da Calahonda — implementado, com 8 testes unitários.\nCódigo completo no GitHub (link na bio).\n\nQual métrica de risco você mais usa no dia a dia? 👇\n\n#Python #QuantFinance #RiskManagement #DataScience #B3 #VaR #MonteCarlo`,
  3: `Todo mundo diz "faça projetos para o portfólio."\nPoucos explicam o que isso realmente significa.\n\nDeixa eu ser direto sobre o que construí e o que aprendi:\n\nO que está implementado hoje (você pode inspecionar o código):\n✔ Landing page responsiva — mobile, tablet, desktop\n✔ Design system com CSS Custom Properties — sem Bootstrap\n✔ Scorecard interativo com Intersection Observer API\n✔ Progress bar de leitura com scroll listener passive\n✔ Clipboard API para 3 posts LinkedIn prontos\n✔ Toggle PT/EN sem biblioteca — JS puro, zero reload\n✔ CSS Grid layout complexo — fr units + gap\n✔ Módulo VaR em Python — 3 métodos, CVaR e 8 testes\n\nO que ainda é roadmap (honestamente):\n○ Dashboard com dados reais da B3\n○ Backend FastAPI + banco de dados\n○ Machine Learning — Fase 3\n\nDistinguir o que funciona do que é plano foi o feedback mais valioso que recebi.\n\nO que nenhuma aula ensinou:\n1. CSS Grid em 5 linhas resolve o que eu fazia em 50 com float\n2. Intersection Observer é mais eficiente que scroll listener\n3. Projetos sem domínio de negócio são genéricos\n4. README bilíngue vale mais que 10 projetos mal documentados\n\nCódigo no GitHub: github.com/robertochiocca\n\n#PortfolioDeDesenvolvedor #HTML #CSS #JavaScript #DataScience #PUC #EstágioDev`
};

function copyPost(n) {
  navigator.clipboard.writeText(posts[n]).then(() => {
    const el = document.getElementById('ok' + n);
    el.style.display = 'block';
    setTimeout(() => el.style.display = 'none', 4000);
  }).catch(() => {
    // Clipboard indisponível (http:// ou permissão negada): seleção manual
    window.prompt('Copie o texto abaixo (Ctrl+C):', posts[n]);
  });
}
