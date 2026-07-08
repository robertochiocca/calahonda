<div align="center">

# ⚡ Calahonda

**Finanças quantitativas para o mercado brasileiro (B3) — um projeto de portfólio.**

[![CI](https://github.com/robertochiocca/calahonda/actions/workflows/ci.yml/badge.svg)](https://github.com/robertochiocca/calahonda/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Testes](https://img.shields.io/badge/testes-8%2F8-brightgreen.svg)](tests/test_var.py)
[![Licença: MIT](https://img.shields.io/badge/licen%C3%A7a-MIT-green.svg)](LICENSE)

🇧🇷 [Português](#-versão-em-português) · 🇺🇸 [English](#-english-version)

</div>

> ℹ️ **Conceito / projeto de portfólio — não é uma empresa.** Este repositório estuda como uma plataforma de análise quantitativa focada em gestoras independentes brasileiras *poderia* funcionar. O front-end e o módulo de VaR em Python estão implementados e testados; todo o resto é roadmap explicitamente sinalizado.

---

## 🇧🇷 Versão em Português

**Calahonda** é um projeto de portfólio que explora análise quantitativa aplicada ao mercado brasileiro. A hipótese: gestoras independentes menores raramente pagam pelas ferramentas quant internacionais (Bloomberg, FactSet, na casa dos milhares de dólares/mês), então há espaço para estudar uma alternativa focada na B3, em português.

Comecei pela peça mais concreta e testável de qualquer plataforma de risco: **o cálculo de VaR em Python**.

### 📊 O que já funciona (código real)

Um **módulo de Value at Risk (VaR) e Conditional VaR (CVaR)** em Python, com três métodos clássicos da indústria:

| Método | Ideia central |
|---|---|
| 📊 **Histórico** | quantil da distribuição empírica dos retornos |
| 📈 **Paramétrico** | variância-covariância (fórmula fechada, normal) |
| 🎲 **Monte Carlo** | 10.000 simulações com semente reprodutível |

Mais **8 testes unitários** — incluindo um que valida o resultado contra a teoria da distribuição normal (VaR 95% ≈ 1.645·σ). Roda 100% offline (dados sintéticos) ou com dados reais da B3 via `yfinance`.

**Stack:** Python 3.10+ · NumPy · pandas · SciPy · pytest

### 🚀 Como rodar

```bash
git clone https://github.com/robertochiocca/calahonda.git
cd calahonda
pip install -r requirements.txt

python examples/example_var.py   # exemplo completo
pytest                           # 8 testes
```

Saída real (carteira de R$1M em PETR4 + VALE3 + ITUB4, VaR 95% em 21 dias úteis):

```
              VaR %  CVaR %     VaR R$    CVaR R$
Método
historical   11.309  13.996  113086.53  139963.91
parametric   10.502  13.309  105021.66  133088.30
monte_carlo  10.366  13.074  103655.94  130742.66

Retorno anualizado :   6.55%
Volatilidade anual :  23.27%
Sharpe (rf=0)      :    0.28
```

### 🐍 O código do VaR (Monte Carlo)

O coração do módulo — valida cada premissa de forma explícita e usa NumPy vetorizado:

```python
def monte_carlo_var(returns, confidence=0.95, horizon_days=1,
                    n_sims=10_000, seed=42):
    """VaR e CVaR por simulação de Monte Carlo.

    Estima média e desvio dos retornos, simula milhares de cenários no
    horizonte e extrai o quantil das perdas acumuladas.
    """
    r = _clean_returns(returns, confidence, horizon_days)
    mu, sigma = r.mean(), r.std(ddof=1)
    rng = np.random.default_rng(seed)

    # Simula n_sims cenários e soma os retornos ao longo do horizonte
    sims = rng.normal(mu, sigma, size=(n_sims, horizon_days))
    losses = -sims.sum(axis=1)

    # VaR = quantil das perdas; CVaR = média das perdas além do VaR
    var = float(np.quantile(losses, confidence))
    cvar = _cvar_from_losses(losses, var)
    return RiskEstimate(var, cvar)
```

O módulo completo (em [`calahonda_var/`](calahonda_var/)) tem ainda os métodos **histórico** e **paramétrico**, Sharpe anualizado, carregamento de dados (yfinance + fallback sintético) e os [8 testes](tests/test_var.py).

### 📁 Estrutura do repositório

```
calahonda/
├── index.html · style.css · script.js   ← landing page (HTML/CSS/JS puro)
├── calahonda_var/                       ← núcleo quantitativo em Python
│   ├── var.py                           ← VaR (3 métodos) · CVaR · Sharpe
│   └── data.py                          ← yfinance + fallback sintético
├── tests/test_var.py                    ← 8 testes (pytest)
├── examples/example_var.py              ← exemplo completo
├── requirements.txt · pyproject.toml
└── .github/workflows/ci.yml             ← CI: testes em Python 3.10–3.12
```

### 📌 Estado do projeto — honesto

| Componente | Status | Tecnologia |
|---|---|---|
| Módulo VaR (3 métodos + CVaR) | ✅ **Implementado e testado** | Python · NumPy · pandas · SciPy |
| Landing page / portfólio | ✅ Ao vivo | HTML + CSS + JS puro |
| Toggle PT/EN | ✅ Ao vivo | JavaScript puro |
| CI (testes automáticos) | ✅ Ao vivo | GitHub Actions |
| Dashboard | 🔄 Próximo passo | Streamlit + yfinance |
| Backend API | ⬜ Roadmap | FastAPI + PostgreSQL |
| Modelos ML (LSTM, etc.) | ⬜ Roadmap | scikit-learn / TensorFlow |

### 🎯 O contexto (com fontes)

- Centenas de fundos de ações operam na B3 — ver dados da [B3](https://www.b3.com.br) e da [ANBIMA](https://www.anbima.com.br)
- Mais de 5 milhões de investidores PF na B3 — [B3, 2024](https://www.b3.com.br)
- Bloomberg Terminal custa na casa de milhares de dólares/mês — [Bloomberg](https://www.bloomberg.com/professional/products/bloomberg-terminal/)
- Regulação relevante: [CVM](https://www.cvm.gov.br)

> Estes números são para dar contexto ao problema, não afirmações de mercado validadas. Um projeto honesto trata cada premissa como hipótese a validar.

### 📍 A origem do nome

O nome vem do **Sítio de Calahonda**, na costa de Málaga, sul da Espanha — a cidade de onde minha família emigrou. Assim como uma enseada abrigada (o que *calahonda* evoca: uma "cala honda", ou angra profunda), o projeto foi pensado como um porto seguro: um lugar onde decisões se ancoram em dados sólidos.

> *Sitio de Calahonda · Mijas, Málaga · Andalucía, España*

### 💡 O que este projeto me ensinou

1. O núcleo de finanças quantitativas é estatística aplicada — VaR, CVaR e Monte Carlo são, no fundo, distribuições e quantis
2. Escrever testes que validam contra a teoria (ex.: VaR paramétrico ≈ 1.645σ) é o que separa "rodou" de "está correto"
3. Honestidade técnica é credibilidade — separar o que funciona do que é roadmap foi o feedback mais valioso que recebi
4. Domínio de negócio amplifica o valor técnico: Data Science sem contexto é commodity

---
---

## 🇺🇸 English Version

**Calahonda** is a portfolio project exploring quantitative analysis for the Brazilian market. The hypothesis: smaller independent managers rarely pay for international quant tools (Bloomberg, FactSet, in the thousands of dollars/month), so there's room to study a B3-focused alternative in Portuguese.

I started with the most concrete, testable piece of any risk platform: **VaR calculation in Python**.

### 📊 What already works (real code)

A **Value at Risk (VaR) and Conditional VaR (CVaR)** module in Python, with three industry-standard methods:

| Method | Core idea |
|---|---|
| 📊 **Historical** | quantile of the empirical return distribution |
| 📈 **Parametric** | variance-covariance (closed-form, normal) |
| 🎲 **Monte Carlo** | 10,000 simulations with a reproducible seed |

Plus **8 unit tests** — including one that validates the result against normal-distribution theory (95% VaR ≈ 1.645·σ). Runs 100% offline (synthetic data) or with real B3 data via `yfinance`.

**Stack:** Python 3.10+ · NumPy · pandas · SciPy · pytest

### 🚀 Quick start

```bash
git clone https://github.com/robertochiocca/calahonda.git
cd calahonda
pip install -r requirements.txt

python examples/example_var.py   # full example
pytest                           # 8 tests
```

Real output (R$1M portfolio in PETR4 + VALE3 + ITUB4, 95% VaR over 21 trading days):

```
              VaR %  CVaR %     VaR R$    CVaR R$
Method
historical   11.309  13.996  113086.53  139963.91
parametric   10.502  13.309  105021.66  133088.30
monte_carlo  10.366  13.074  103655.94  130742.66

Annualized return  :   6.55%
Annualized vol     :  23.27%
Sharpe (rf=0)      :    0.28
```

### 🐍 The VaR code (Monte Carlo)

The heart of the module — validates every assumption explicitly and uses vectorized NumPy:

```python
def monte_carlo_var(returns, confidence=0.95, horizon_days=1,
                    n_sims=10_000, seed=42):
    """VaR and CVaR via Monte Carlo simulation.

    Estimates mean and std of returns, simulates thousands of scenarios
    over the horizon, and extracts the quantile of cumulative losses.
    """
    r = _clean_returns(returns, confidence, horizon_days)
    mu, sigma = r.mean(), r.std(ddof=1)
    rng = np.random.default_rng(seed)

    # Simulate n_sims scenarios and sum returns over the horizon
    sims = rng.normal(mu, sigma, size=(n_sims, horizon_days))
    losses = -sims.sum(axis=1)

    # VaR = loss quantile; CVaR = mean of losses beyond the VaR
    var = float(np.quantile(losses, confidence))
    cvar = _cvar_from_losses(losses, var)
    return RiskEstimate(var, cvar)
```

The full module (in [`calahonda_var/`](calahonda_var/)) also includes the **historical** and **parametric** methods, annualized Sharpe, data loading (yfinance + synthetic fallback) and the [8 tests](tests/test_var.py).

### 📁 Repository structure

```
calahonda/
├── index.html · style.css · script.js   ← landing page (pure HTML/CSS/JS)
├── calahonda_var/                       ← Python quantitative core
│   ├── var.py                           ← VaR (3 methods) · CVaR · Sharpe
│   └── data.py                          ← yfinance + synthetic fallback
├── tests/test_var.py                    ← 8 tests (pytest)
├── examples/example_var.py              ← full example
├── requirements.txt · pyproject.toml
└── .github/workflows/ci.yml             ← CI: tests on Python 3.10–3.12
```

### 📌 Project status — honest

| Component | Status | Technology |
|---|---|---|
| VaR module (3 methods + CVaR) | ✅ **Implemented and tested** | Python · NumPy · pandas · SciPy |
| Landing page / portfolio | ✅ Live | Pure HTML + CSS + JS |
| PT/EN toggle | ✅ Live | Pure JavaScript |
| CI (automated tests) | ✅ Live | GitHub Actions |
| Dashboard | 🔄 Next step | Streamlit + yfinance |
| Backend API | ⬜ Roadmap | FastAPI + PostgreSQL |
| ML models (LSTM, etc.) | ⬜ Roadmap | scikit-learn / TensorFlow |

### 🎯 The context (with sources)

- Hundreds of equity funds operate on B3 — see [B3](https://www.b3.com.br) and [ANBIMA](https://www.anbima.com.br) data
- Over 5 million retail investors on B3 — [B3, 2024](https://www.b3.com.br)
- Bloomberg Terminal costs in the thousands of dollars/month — [Bloomberg](https://www.bloomberg.com/professional/products/bloomberg-terminal/)
- Relevant regulation: [CVM](https://www.cvm.gov.br)

> These figures are context for the problem, not validated market claims. An honest project treats each assumption as a hypothesis to test.

### 📍 The origin of the name

The name comes from **Sitio de Calahonda**, on the Málaga coast in southern Spain — the town my family emigrated from. Like a sheltered cove (what *calahonda* evokes: a "cala honda", or deep inlet), the project was conceived as a safe harbor: a place where decisions anchor in solid data.

> *Sitio de Calahonda · Mijas, Málaga · Andalucía, Spain*

### 💡 What this project taught me

1. The core of quant finance is applied statistics — VaR, CVaR and Monte Carlo are, at heart, distributions and quantiles
2. Writing tests that validate against theory (e.g. parametric VaR ≈ 1.645σ) is what separates "it ran" from "it's correct"
3. Technical honesty is credibility — separating what works from what's roadmap was the most valuable feedback I received
4. Business domain amplifies technical value: Data Science without context is commodity

---

<div align="center">

**Roberto Chiocca** · [github.com/robertochiocca](https://github.com/robertochiocca)

*Fins educacionais apenas — não é recomendação de investimento. · Educational purposes only — not investment advice.*

Licenciado sob a [Licença MIT](LICENSE) · Licensed under the [MIT License](LICENSE)

</div>
