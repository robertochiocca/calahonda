"""Calahonda — dashboard de risco quantitativo (Streamlit + Plotly).

Uso::

    pip install -r requirements.txt
    streamlit run streamlit_app.py

Com internet, usa preços reais da B3 via yfinance; sem internet, cai
automaticamente nos dados sintéticos do módulo. Aceita upload de
carteira em CSV, Excel (.xlsx) ou JSON com colunas ``ticker,peso``
(ver ``examples/carteira_exemplo.csv``).
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from calahonda_var import (
    align_with_benchmark,
    annualized_return,
    annualized_volatility,
    backtest_var,
    beta_to_benchmark,
    correlation_matrix,
    drawdown_series,
    equity_curve,
    generate_pdf_report,
    load_benchmark,
    load_portfolio_file,
    load_returns,
    max_drawdown,
    max_sharpe_weights,
    metrics_report,
    min_variance_weights,
    monte_carlo_var,
    portfolio_returns,
    portfolio_volatility,
    sharpe_ratio,
    stress_test,
    var_report,
    volatility_shock_var,
    weighted_shock,
)
from calahonda_var.theme import AMBER, BLUE, LIME, PLOTLY_LAYOUT, RED

# ---------------------------------------------------------------- constantes
N_MONTE_CARLO = 10_000
TRADING_DAYS = 252
CACHE_TTL_SECONDS = 3600
RANDOM_SEED = 42
VOL_SHOCK_MULTIPLIER = 2.0
BACKTEST_WINDOW = 252
DEFAULT_PORTFOLIO_VALUE = 1_000_000.0
DEFAULT_HORIZON_DAYS = 21
CONFIDENCE_LEVELS = (0.90, 0.95, 0.99)
HISTOGRAM_BINS = 60
MAX_SAVED_PORTFOLIOS = 4  # limite da paleta de comparação (cores fixas)

# Limiares de leitura do VaR como fração da carteira no horizonte:
# abaixo do primeiro = risco baixo; acima do segundo = risco alto.
VAR_THRESHOLD_LOW = 0.08
VAR_THRESHOLD_HIGH = 0.15

DEFAULT_UNIVERSE = (
    "PETR4.SA",
    "VALE3.SA",
    "ITUB4.SA",
    "BBAS3.SA",
    "WEGE3.SA",
    "B3SA3.SA",
)

HOW_TO_USE = """
1. Escolha os ativos (ou envie um arquivo CSV/XLSX/JSON)
2. Ajuste os pesos de cada ativo
3. Defina valor, confiança e horizonte
4. Explore as abas e baixe o relatório em PDF
"""


@dataclass(frozen=True)
class PortfolioConfig:
    """Parâmetros escolhidos pelo usuário na sidebar."""

    tickers: tuple[str, ...]
    weights: np.ndarray
    value: float
    confidence: float
    horizon_days: int


# ---------------------------------------------------------------- dados
@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Baixando dados da B3…")
def fetch_returns(tickers: tuple[str, ...]) -> tuple[pd.DataFrame, bool]:
    """Retornos dos ativos + flag indicando se os dados são reais."""
    try:
        return load_returns(tickers, fallback=False), True
    except Exception:
        return load_returns(tickers, fallback=True), False


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_benchmark() -> pd.Series:
    """Retornos do Ibovespa (ou benchmark sintético, offline)."""
    return load_benchmark()


# ---------------------------------------------------------------- helpers
def risk_level(var_fraction: float) -> tuple[str, str]:
    """Classifica o VaR (fração da carteira) em nível e cor do Streamlit."""
    if var_fraction < VAR_THRESHOLD_LOW:
        return "risco baixo", "green"
    if var_fraction < VAR_THRESHOLD_HIGH:
        return "risco moderado", "orange"
    return "risco alto", "red"


def themed_figure(height: int = 340) -> go.Figure:
    """Figura Plotly vazia com o tema escuro da Calahonda."""
    figure = go.Figure()
    figure.update_layout(**PLOTLY_LAYOUT, height=height)
    return figure


def simulate_terminal_pnl(
    portfolio: pd.Series, horizon_days: int, n_sims: int = N_MONTE_CARLO
) -> np.ndarray:
    """P&L percentual simulado no horizonte (Monte Carlo, semente fixa)."""
    rng = np.random.default_rng(RANDOM_SEED)
    steps = rng.normal(
        portfolio.mean(), portfolio.std(ddof=1), size=(n_sims, horizon_days)
    )
    return steps.sum(axis=1) * 100.0


def monte_carlo_figure(
    terminal_pnl: np.ndarray, var_fraction: float, cvar_fraction: float
) -> go.Figure:
    """Histograma interativo do P&L simulado com linhas de VaR e CVaR."""
    figure = themed_figure(height=360)
    figure.add_trace(
        go.Histogram(
            x=terminal_pnl,
            nbinsx=HISTOGRAM_BINS,
            marker_color=LIME,
            opacity=0.85,
            hovertemplate="P&L %{x:.1f}%%: %{y} cenários<extra></extra>",
        )
    )
    figure.add_vline(
        x=-var_fraction * 100,
        line_color=AMBER,
        line_width=2,
        annotation_text=f"VaR {var_fraction:.1%}",
        annotation_font_color=AMBER,
    )
    figure.add_vline(
        x=-cvar_fraction * 100,
        line_color=RED,
        line_width=2,
        annotation_text=f"CVaR {cvar_fraction:.1%}",
        annotation_font_color=RED,
        annotation_position="bottom left",
    )
    figure.update_layout(
        xaxis_title="P&L acumulado no horizonte (%)",
        yaxis_title="Cenários",
        showlegend=False,
    )
    return figure


def equity_drawdown_figure(portfolio: pd.Series) -> go.Figure:
    """Curva de patrimônio (base 100) e drawdown, com hover e zoom."""
    curve = equity_curve(portfolio, initial=100.0)
    drawdown = drawdown_series(portfolio) * 100.0
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.06,
    )
    figure.add_trace(
        go.Scatter(
            x=curve.index,
            y=curve,
            line={"color": LIME, "width": 1.6},
            name="Patrimônio",
            hovertemplate="%{x|%d/%m/%Y}: %{y:.1f}<extra>Patrimônio</extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown,
            fill="tozeroy",
            line={"color": RED, "width": 1.0},
            name="Drawdown",
            hovertemplate="%{x|%d/%m/%Y}: %{y:.1f}%<extra>Drawdown</extra>",
        ),
        row=2,
        col=1,
    )
    figure.update_layout(**PLOTLY_LAYOUT, height=420, showlegend=False)
    figure.update_yaxes(title_text="Base 100", row=1, col=1)
    figure.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    return figure


def benchmark_figure(aligned: pd.DataFrame) -> go.Figure:
    """Carteira vs. Ibovespa em base 100, no período em comum."""
    figure = themed_figure(height=360)
    base100 = (1.0 + aligned).cumprod() * 100.0
    for column, color, label in (
        ("portfolio", LIME, "Sua carteira"),
        ("benchmark", BLUE, "Ibovespa"),
    ):
        figure.add_trace(
            go.Scatter(
                x=base100.index,
                y=base100[column],
                line={"color": color, "width": 1.6},
                name=label,
                hovertemplate="%{x|%d/%m/%Y}: %{y:.1f}<extra>" + label + "</extra>",
            )
        )
    figure.update_layout(
        yaxis_title="Base 100",
        legend={"orientation": "h", "y": 1.08},
    )
    return figure


# ---------------------------------------------------------------- sidebar
def sidebar_inputs() -> PortfolioConfig:
    """Coleta a carteira e os parâmetros de risco na sidebar."""
    st.sidebar.title("⚡ Calahonda")
    st.sidebar.caption(
        "Risco quantitativo para a B3 — projeto educacional, "
        "não é recomendação de investimento."
    )
    with st.sidebar.expander("Como usar"):
        st.markdown(HOW_TO_USE)

    uploaded = st.sidebar.file_uploader(
        "Importar carteira (ticker,peso)",
        type=["csv", "xlsx", "json"],
        help="Exemplo em examples/carteira_exemplo.csv",
    )
    if uploaded is not None:
        try:
            tickers, weights = load_portfolio_file(uploaded, name=uploaded.name)
        except ValueError as error:
            st.sidebar.error(str(error))
            st.stop()
        st.sidebar.success(f"Carteira importada: {len(tickers)} ativos")
    else:
        chosen = st.sidebar.multiselect(
            "Ativos (B3)", DEFAULT_UNIVERSE, default=DEFAULT_UNIVERSE[:3]
        )
        if len(chosen) < 2:
            st.sidebar.warning("Escolha pelo menos 2 ativos.")
            st.stop()
        raw_weights = [
            st.sidebar.slider(
                f"Peso {ticker.replace('.SA', '')} (%)",
                0,
                100,
                round(100 / len(chosen)),
                5,
            )
            for ticker in chosen
        ]
        if sum(raw_weights) == 0:
            st.sidebar.warning("A soma dos pesos precisa ser maior que zero.")
            st.stop()
        tickers = tuple(chosen)
        weights = np.array(raw_weights, dtype=float) / sum(raw_weights)

    value = st.sidebar.number_input(
        "Valor da carteira (R$)",
        1_000.0,
        1e9,
        DEFAULT_PORTFOLIO_VALUE,
        100_000.0,
        format="%.0f",
    )
    confidence = st.sidebar.select_slider(
        "Confiança do VaR",
        CONFIDENCE_LEVELS,
        value=0.95,
        format_func=lambda level: f"{level:.0%}",
    )
    horizon_days = st.sidebar.number_input(
        "Horizonte (dias úteis)", 1, TRADING_DAYS, DEFAULT_HORIZON_DAYS
    )
    return PortfolioConfig(tickers, weights, value, confidence, int(horizon_days))


# ---------------------------------------------------------------- abas
def render_header(
    config: PortfolioConfig, portfolio: pd.Series, var_mc: float, cvar_mc: float
) -> None:
    """Título e faixa de métricas, com o VaR classificado por cor."""
    st.title("Dashboard de risco da carteira")
    st.caption(
        " · ".join(
            f"{ticker.replace('.SA', '')} {weight:.0%}"
            for ticker, weight in zip(config.tickers, config.weights, strict=True)
        )
    )
    level_label, level_color = risk_level(var_mc)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(
        f"VaR {config.confidence:.0%} · {config.horizon_days}d",
        f"R$ {var_mc * config.value:,.0f}",
        f"-{var_mc:.1%} da carteira",
        delta_color="off",
    )
    col1.markdown(f":{level_color}[● {level_label}]")
    col2.metric(
        "CVaR (Expected Shortfall)",
        f"R$ {cvar_mc * config.value:,.0f}",
        f"-{cvar_mc:.1%}",
        delta_color="off",
    )
    col3.metric("Volatilidade anual", f"{annualized_volatility(portfolio):.1%}")
    col4.metric("Sharpe (rf=0)", f"{sharpe_ratio(portfolio):.2f}")
    col5.metric("Máx. drawdown", f"{max_drawdown(portfolio):.1%}")


def render_risk_tab(
    config: PortfolioConfig, portfolio: pd.Series, var_mc: float, cvar_mc: float
) -> None:
    left, right = st.columns([1, 1.3])
    with left:
        st.subheader("VaR e CVaR pelos 3 métodos")
        st.dataframe(
            var_report(portfolio, config.value, config.confidence, config.horizon_days),
            use_container_width=True,
        )
        st.caption("Convenção: números positivos = perda potencial.")
    with right:
        terminal_pnl = simulate_terminal_pnl(portfolio, config.horizon_days)
        st.plotly_chart(
            monte_carlo_figure(terminal_pnl, var_mc, cvar_mc),
            use_container_width=True,
        )
        st.caption(
            f"{N_MONTE_CARLO:,} cenários de Monte Carlo — arraste para dar "
            "zoom, passe o mouse para inspecionar.".replace(",", ".")
        )


def render_performance_tab(portfolio: pd.Series, returns: pd.DataFrame) -> None:
    st.subheader("Patrimônio e drawdown")
    st.plotly_chart(equity_drawdown_figure(portfolio), use_container_width=True)
    st.metric("Retorno anualizado", f"{annualized_return(portfolio):.2%}")

    st.subheader("Sua carteira vs. Ibovespa")
    benchmark = fetch_benchmark()
    try:
        aligned = align_with_benchmark(portfolio, benchmark)
        st.plotly_chart(benchmark_figure(aligned), use_container_width=True)
        beta = beta_to_benchmark(portfolio, benchmark)
        correlation = aligned["portfolio"].corr(aligned["benchmark"])
        col1, col2, col3 = st.columns(3)
        col1.metric("Beta vs. Ibovespa", f"{beta:.2f}")
        col2.metric("Correlação", f"{correlation:.2f}")
        col3.metric("Sharpe do Ibovespa", f"{sharpe_ratio(aligned['benchmark']):.2f}")
        st.caption(f"Benchmark: {benchmark.name} · período em comum com a carteira.")
    except ValueError as error:
        st.info(f"Comparação indisponível: {error}")

    st.subheader("Métricas avançadas")
    st.dataframe(metrics_report(portfolio, benchmark), use_container_width=True)
    st.caption(
        "Sortino, Calmar, Omega, Treynor, Jensen Alpha, Information Ratio "
        "e tracking error — valores '—' são indefinidos para a amostra."
    )

    st.subheader("Correlação entre os ativos")
    st.dataframe(correlation_matrix(returns).round(3), use_container_width=True)


def render_stress_tab(config: PortfolioConfig, portfolio: pd.Series) -> None:
    left, right = st.columns(2)
    with left:
        st.subheader("Cenários históricos")
        beta = st.slider("Beta da carteira vs. Ibovespa", 0.2, 2.0, 1.0, 0.1)
        st.dataframe(stress_test(config.value, beta=beta), use_container_width=True)
        shock = volatility_shock_var(
            portfolio, VOL_SHOCK_MULTIPLIER, config.confidence, config.horizon_days
        )
        st.metric(
            f"VaR com volatilidade {VOL_SHOCK_MULTIPLIER:.0f}x (regime de crise)",
            f"R$ {shock.var * config.value:,.0f}",
            f"-{shock.var:.1%}",
            delta_color="off",
        )
    with right:
        st.subheader("Backtest de VaR (Kupiec)")
        try:
            result = backtest_var(
                portfolio, confidence=config.confidence, window=BACKTEST_WINDOW
            )
            st.metric(
                "Violações observadas vs. esperadas",
                f"{result.violations} / {result.expected_violations:.0f}",
                f"taxa {result.violation_rate:.1%}",
                delta_color="off",
            )
            verdict = "APROVADO" if result.approved else "REPROVADO"
            verdict_color = "green" if result.approved else "red"
            st.metric("Teste de Kupiec (p-valor)", f"{result.kupiec_pvalue:.3f}")
            st.markdown(f":{verdict_color}[● {verdict}]")
            st.caption(
                f"{result.observations} dias out-of-sample, janela móvel de "
                f"{BACKTEST_WINDOW} pregões — o VaR de cada dia usa só o passado."
            )
        except ValueError as error:
            st.info(f"Histórico insuficiente para backtest: {error}")

    st.subheader("Cenário personalizado")
    st.caption(
        "Defina um choque por ativo (negativo = queda) e veja o impacto "
        "combinado na carteira — ex.: Petrobras -15%, Vale -8%."
    )
    editor_default = pd.DataFrame(
        {
            "Ativo": [t.replace(".SA", "") for t in config.tickers],
            "Peso %": np.round(config.weights * 100, 1),
            "Choque %": np.full(len(config.tickers), -10.0),
        }
    )
    edited = st.data_editor(
        editor_default,
        disabled=["Ativo", "Peso %"],
        hide_index=True,
        use_container_width=True,
        key="custom-scenario",
    )
    shock = weighted_shock(config.weights, edited["Choque %"].to_numpy() / 100.0)
    st.metric(
        "Impacto do cenário na carteira",
        f"R$ {shock * config.value:,.0f}",
        f"{shock:.1%}",
    )


def render_optimization_tab(config: PortfolioConfig, returns: pd.DataFrame) -> None:
    st.subheader("Carteiras ótimas (long-only, Markowitz)")
    comparison = pd.DataFrame(
        {
            "Sua carteira": config.weights,
            "Mínima variância": min_variance_weights(returns).to_numpy(),
            "Máximo Sharpe": max_sharpe_weights(returns).to_numpy(),
        },
        index=[ticker.replace(".SA", "") for ticker in returns.columns],
    )
    st.dataframe(comparison.style.format("{:.1%}"), use_container_width=True)

    def portfolio_stats(weights: np.ndarray) -> tuple[float, float]:
        portfolio_series = portfolio_returns(returns, weights)
        return (
            portfolio_volatility(returns, weights),
            sharpe_ratio(portfolio_series),
        )

    stats_rows = [portfolio_stats(comparison[c].to_numpy()) for c in comparison]
    st.dataframe(
        pd.DataFrame(
            stats_rows,
            index=comparison.columns,
            columns=["Volatilidade anual", "Sharpe"],
        ).style.format({"Volatilidade anual": "{:.2%}", "Sharpe": "{:.2f}"}),
        use_container_width=True,
    )
    st.caption(
        "Otimização via SciPy (SLSQP): sem alavancagem e sem venda a "
        "descoberto. Restrições de liquidez e drawdown são roadmap."
    )


def render_portfolios_tab(config: PortfolioConfig) -> None:
    """Salva a carteira atual e compara a evolução entre as salvas."""
    st.subheader("Salvar e comparar carteiras")
    saved: dict = st.session_state.setdefault("saved_portfolios", {})

    name_col, save_col, clear_col = st.columns([2, 1, 1])
    default_name = f"Carteira {chr(65 + (len(saved) % 26))}"
    name = name_col.text_input("Nome da carteira", value=default_name)
    if save_col.button("Salvar carteira atual", use_container_width=True):
        if len(saved) >= MAX_SAVED_PORTFOLIOS and name not in saved:
            st.warning(f"Máximo de {MAX_SAVED_PORTFOLIOS} carteiras salvas.")
        else:
            saved[name] = {
                "tickers": list(config.tickers),
                "weights": [float(w) for w in config.weights],
                "value": float(config.value),
            }
            st.success(
                f"'{name}' salva. Monte outra carteira na sidebar e salve "
                "de novo para comparar."
            )
    if clear_col.button("Limpar salvas", use_container_width=True):
        saved.clear()

    imported = st.file_uploader(
        "Importar carteiras salvas (JSON)", type="json", key="import-saved"
    )
    if imported is not None:
        try:
            data = json.load(imported)
            valid = {
                str(label): entry
                for label, entry in data.items()
                if isinstance(entry, dict)
                and {"tickers", "weights", "value"} <= entry.keys()
            }
            saved.update(dict(list(valid.items())[:MAX_SAVED_PORTFOLIOS]))
        except (json.JSONDecodeError, AttributeError):
            st.error("JSON inválido: use o arquivo exportado por este app.")

    if not saved:
        st.info(
            "Salve a carteira atual, monte outra na sidebar e salve de novo — "
            "aí dá para comparar a evolução, o VaR e o Sharpe entre elas."
        )
        return

    figure = themed_figure(height=380)
    palette = (LIME, BLUE, AMBER, RED)
    comparison: dict[str, dict[str, str]] = {}
    for (label, entry), color in zip(saved.items(), palette, strict=False):
        returns, _ = fetch_returns(tuple(entry["tickers"]))
        series = portfolio_returns(returns, np.asarray(entry["weights"]))
        curve = equity_curve(series, initial=100.0)
        figure.add_trace(
            go.Scatter(
                x=curve.index,
                y=curve,
                name=label,
                line={"color": color, "width": 1.6},
                hovertemplate="%{x|%d/%m/%Y}: %{y:.1f}<extra>" + label + "</extra>",
            )
        )
        var_est, _ = monte_carlo_var(
            series, config.confidence, config.horizon_days, N_MONTE_CARLO, RANDOM_SEED
        )
        comparison[label] = {
            f"VaR {config.confidence:.0%} ({config.horizon_days}d)": f"{var_est:.1%}",
            "Volatilidade anual": f"{annualized_volatility(series):.1%}",
            "Sharpe": f"{sharpe_ratio(series):.2f}",
            "Máx. drawdown": f"{max_drawdown(series):.1%}",
        }
    figure.update_layout(yaxis_title="Base 100", legend={"orientation": "h", "y": 1.08})
    st.plotly_chart(figure, use_container_width=True)
    st.dataframe(pd.DataFrame(comparison), use_container_width=True)
    st.download_button(
        "Exportar carteiras (JSON)",
        json.dumps(saved, indent=2, ensure_ascii=False),
        file_name="carteiras_calahonda.json",
        mime="application/json",
    )


def render_report_tab(config: PortfolioConfig, portfolio: pd.Series) -> None:
    st.subheader("Relatório completo em PDF")
    st.write(
        "Três páginas: métricas + stress testing, curva de patrimônio "
        "com drawdown, e distribuição de Monte Carlo."
    )
    if st.button("Gerar relatório", type="primary"):
        buffer = io.BytesIO()
        generate_pdf_report(
            portfolio,
            buffer,
            portfolio_value=config.value,
            confidence=config.confidence,
            horizon_days=config.horizon_days,
        )
        st.download_button(
            "Baixar relatorio_carteira.pdf",
            buffer.getvalue(),
            file_name="relatorio_carteira.pdf",
            mime="application/pdf",
        )


# ---------------------------------------------------------------- main
def main() -> None:
    st.set_page_config(
        page_title="Calahonda — Risco Quantitativo", page_icon="⚡", layout="wide"
    )
    config = sidebar_inputs()
    returns, is_real = fetch_returns(config.tickers)
    portfolio = portfolio_returns(returns, config.weights)
    source = "Dados reais da B3 (yfinance)" if is_real else "Dados sintéticos (offline)"
    st.sidebar.markdown(f"**Fonte:** {source} · {len(portfolio)} pregões")

    var_mc, cvar_mc = monte_carlo_var(
        portfolio, config.confidence, config.horizon_days, N_MONTE_CARLO, RANDOM_SEED
    )
    render_header(config, portfolio, var_mc, cvar_mc)

    tab_risk, tab_perf, tab_stress, tab_opt, tab_saved, tab_pdf = st.tabs(
        [
            "Risco (VaR)",
            "Performance",
            "Stress & Backtest",
            "Otimização",
            "Carteiras",
            "Relatório PDF",
        ]
    )
    with tab_risk:
        render_risk_tab(config, portfolio, var_mc, cvar_mc)
    with tab_perf:
        render_performance_tab(portfolio, returns)
    with tab_stress:
        render_stress_tab(config, portfolio)
    with tab_opt:
        render_optimization_tab(config, returns)
    with tab_saved:
        render_portfolios_tab(config)
    with tab_pdf:
        render_report_tab(config, portfolio)


if __name__ == "__main__":
    main()
