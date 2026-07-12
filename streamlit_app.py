"""Calahonda — dashboard de risco quantitativo (Streamlit).

Uso::

    pip install -r requirements.txt
    streamlit run streamlit_app.py

Com internet, usa preços reais da B3 via yfinance; sem internet, cai
automaticamente nos dados sintéticos do módulo. Também aceita upload de
carteira em CSV (colunas ``ticker,peso`` — ver
``examples/carteira_exemplo.csv``).
"""

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from calahonda_var import (
    annualized_return,
    annualized_volatility,
    backtest_var,
    correlation_matrix,
    drawdown_series,
    equity_curve,
    generate_pdf_report,
    load_portfolio_csv,
    load_returns,
    max_drawdown,
    max_sharpe_weights,
    min_variance_weights,
    monte_carlo_var,
    portfolio_returns,
    portfolio_volatility,
    sharpe_ratio,
    stress_test,
    var_report,
    volatility_shock_var,
)

# Paleta do site
LIME, AMBER, RED, MUTED, SURFACE = "#C8F135", "#F5A623", "#F24B4B", "#8892A4", "#0D0F14"

st.set_page_config(page_title="Calahonda — Risco Quantitativo",
                   page_icon="⚡", layout="wide")

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": "#C8CDD8", "axes.edgecolor": "#2A2F3E",
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "grid.color": "#2A2F3E", "axes.grid": True, "axes.axisbelow": True,
})

DEFAULT_UNIVERSE = ["PETR4.SA", "VALE3.SA", "ITUB4.SA",
                    "BBAS3.SA", "WEGE3.SA", "B3SA3.SA"]


@st.cache_data(ttl=3600, show_spinner="Baixando dados da B3…")
def fetch_returns(tickers: tuple) -> tuple[pd.DataFrame, bool]:
    """Retornos + flag indicando se os dados são reais (yfinance)."""
    try:
        return load_returns(tickers, fallback=False), True
    except Exception:
        return load_returns(tickers, fallback=True), False


def fig_style(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


# ---------------- Sidebar: carteira e parâmetros ----------------
st.sidebar.title("⚡ Calahonda")
st.sidebar.caption("Risco quantitativo para a B3 — projeto educacional, "
                   "não é recomendação de investimento.")

uploaded = st.sidebar.file_uploader(
    "Importar carteira (CSV: ticker,peso)", type="csv",
    help="Exemplo em examples/carteira_exemplo.csv",
)
if uploaded is not None:
    try:
        tickers, weights = load_portfolio_csv(uploaded)
        st.sidebar.success(f"Carteira importada: {len(tickers)} ativos")
    except ValueError as err:
        st.sidebar.error(str(err))
        st.stop()
else:
    chosen = st.sidebar.multiselect("Ativos (B3)", DEFAULT_UNIVERSE,
                                    default=DEFAULT_UNIVERSE[:3])
    if len(chosen) < 2:
        st.sidebar.warning("Escolha pelo menos 2 ativos.")
        st.stop()
    raw = [st.sidebar.slider(f"Peso {t.replace('.SA', '')} (%)", 0, 100,
                             round(100 / len(chosen)), 5) for t in chosen]
    if sum(raw) == 0:
        st.sidebar.warning("A soma dos pesos precisa ser maior que zero.")
        st.stop()
    tickers = tuple(chosen)
    weights = np.array(raw, dtype=float) / sum(raw)

value = st.sidebar.number_input("Valor da carteira (R$)", 1_000.0, 1e9,
                                1_000_000.0, 100_000.0, format="%.0f")
confidence = st.sidebar.select_slider("Confiança do VaR",
                                      [0.90, 0.95, 0.99], value=0.95,
                                      format_func=lambda c: f"{c:.0%}")
horizon = st.sidebar.number_input("Horizonte (dias úteis)", 1, 252, 21)

returns, is_real = fetch_returns(tickers)
portfolio = portfolio_returns(returns, weights)

st.sidebar.markdown(
    f"{'🟢 **Dados reais da B3** (yfinance)' if is_real else '🟡 **Dados sintéticos** (sem internet)'}"
    f" · {len(portfolio)} pregões"
)

# ---------------- Cabeçalho ----------------
st.title("Dashboard de risco da carteira")
st.caption(" · ".join(f"{t.replace('.SA', '')} {w:.0%}"
                      for t, w in zip(tickers, weights)))

var_mc, cvar_mc = monte_carlo_var(portfolio, confidence, horizon)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(f"VaR {confidence:.0%} · {horizon}d", f"R$ {var_mc * value:,.0f}",
          f"-{var_mc:.1%} da carteira", delta_color="off")
c2.metric("CVaR (Expected Shortfall)", f"R$ {cvar_mc * value:,.0f}",
          f"-{cvar_mc:.1%}", delta_color="off")
c3.metric("Volatilidade anual", f"{annualized_volatility(portfolio):.1%}")
c4.metric("Sharpe (rf=0)", f"{sharpe_ratio(portfolio):.2f}")
c5.metric("Máx. drawdown", f"{max_drawdown(portfolio):.1%}")

tab_risco, tab_perf, tab_stress, tab_otim, tab_pdf = st.tabs(
    ["🎯 Risco (VaR)", "📈 Performance", "🧪 Stress & Backtest",
     "⚖️ Otimização", "📄 Relatório PDF"]
)

# ---------------- Aba 1: risco ----------------
with tab_risco:
    left, right = st.columns([1, 1.3])
    with left:
        st.subheader("VaR e CVaR pelos 3 métodos")
        st.dataframe(var_report(portfolio, value, confidence, horizon),
                     use_container_width=True)
        st.caption("Convenção: números positivos = perda potencial.")
    with right:
        mu, sigma = portfolio.mean(), portfolio.std(ddof=1)
        terminal = np.random.default_rng(42).normal(
            mu, sigma, size=(10_000, horizon)).sum(axis=1) * 100
        fig, ax = plt.subplots(figsize=(7, 3.4))
        ax.hist(terminal, bins=60, color=LIME, alpha=0.85,
                edgecolor=SURFACE, linewidth=0.5)
        ax.axvline(-var_mc * 100, color=AMBER, linewidth=1.6)
        ax.axvline(-cvar_mc * 100, color=RED, linewidth=1.6)
        ax.text(-var_mc * 100, ax.get_ylim()[1] * 0.95, f" VaR {var_mc:.1%}",
                color=AMBER, fontsize=9)
        ax.text(-cvar_mc * 100, ax.get_ylim()[1] * 0.85, f"CVaR {cvar_mc:.1%} ",
                color=RED, fontsize=9, ha="right")
        ax.set_xlabel(f"P&L simulado em {horizon} dias (%) — 10.000 cenários")
        fig_style(ax)
        st.pyplot(fig, use_container_width=True)

# ---------------- Aba 2: performance ----------------
with tab_perf:
    curve = equity_curve(portfolio, initial=100.0)
    dd = drawdown_series(portfolio) * 100
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1],
                                                "hspace": 0.12})
    ax1.plot(curve.index, curve, color=LIME, linewidth=1.5)
    ax1.set_ylabel("Patrimônio (base 100)")
    ax2.fill_between(dd.index, dd, 0, color=RED, alpha=0.35, linewidth=0)
    ax2.plot(dd.index, dd, color=RED, linewidth=0.9)
    ax2.set_ylabel("Drawdown (%)")
    for ax in (ax1, ax2):
        fig_style(ax)
    st.pyplot(fig, use_container_width=True)
    st.metric("Retorno anualizado", f"{annualized_return(portfolio):.2%}")
    st.subheader("Correlação entre os ativos")
    st.dataframe(correlation_matrix(returns).round(3), use_container_width=True)

# ---------------- Aba 3: stress e backtest ----------------
with tab_stress:
    left, right = st.columns(2)
    with left:
        st.subheader("Cenários históricos")
        beta = st.slider("Beta da carteira vs. Ibovespa", 0.2, 2.0, 1.0, 0.1)
        st.dataframe(stress_test(value, beta=beta), use_container_width=True)
        shock = volatility_shock_var(portfolio, 2.0, confidence, horizon)
        st.metric("VaR com volatilidade 2× (regime de crise)",
                  f"R$ {shock.var * value:,.0f}", f"-{shock.var:.1%}",
                  delta_color="off")
    with right:
        st.subheader("Backtest de VaR (Kupiec)")
        try:
            bt = backtest_var(portfolio, confidence=confidence)
            st.metric("Violações observadas vs. esperadas",
                      f"{bt.violations} / {bt.expected_violations:.0f}",
                      f"taxa {bt.violation_rate:.1%}", delta_color="off")
            verdict = "✅ APROVADO" if bt.approved else "❌ REPROVADO"
            st.metric("Teste de Kupiec (p-valor)",
                      f"{bt.kupiec_pvalue:.3f}", verdict, delta_color="off")
            st.caption(f"{bt.observations} dias out-of-sample, janela móvel "
                       "de 252 pregões — o VaR de cada dia usa só o passado.")
        except ValueError as err:
            st.info(f"Histórico insuficiente para backtest: {err}")

# ---------------- Aba 4: otimização ----------------
with tab_otim:
    st.subheader("Carteiras ótimas (long-only, Markowitz)")
    opt_min = min_variance_weights(returns)
    opt_shp = max_sharpe_weights(returns)
    comp = pd.DataFrame({
        "Sua carteira": weights,
        "Mínima variância": opt_min.to_numpy(),
        "Máximo Sharpe": opt_shp.to_numpy(),
    }, index=[t.replace(".SA", "") for t in returns.columns])
    st.dataframe(comp.style.format("{:.1%}"), use_container_width=True)

    def _stats(w):
        p = portfolio_returns(returns, w)
        return (portfolio_volatility(returns, w), sharpe_ratio(p))

    rows = [_stats(comp[c].to_numpy()) for c in comp.columns]
    st.dataframe(pd.DataFrame(rows, index=comp.columns,
                              columns=["Volatilidade anual", "Sharpe"])
                 .style.format({"Volatilidade anual": "{:.2%}",
                                "Sharpe": "{:.2f}"}),
                 use_container_width=True)
    st.caption("Otimização via SciPy (SLSQP): sem alavancagem e sem venda "
               "a descoberto. Restrições de liquidez e drawdown são roadmap.")

# ---------------- Aba 5: relatório PDF ----------------
with tab_pdf:
    st.subheader("Relatório completo em PDF")
    st.write("Três páginas: métricas + stress testing, curva de patrimônio "
             "com drawdown, e distribuição de Monte Carlo.")
    if st.button("Gerar relatório", type="primary"):
        buffer = io.BytesIO()
        generate_pdf_report(portfolio, buffer, portfolio_value=value,
                            confidence=confidence, horizon_days=horizon)
        st.download_button("⬇️ Baixar relatorio_carteira.pdf", buffer.getvalue(),
                           file_name="relatorio_carteira.pdf",
                           mime="application/pdf")
