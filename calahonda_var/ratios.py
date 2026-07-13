"""Métricas quantitativas avançadas de risco-retorno.

Complementa o Sharpe com os índices que gestoras acompanham:

- **Sortino** — penaliza só a volatilidade das perdas
- **Calmar** — retorno anualizado sobre o máximo drawdown
- **Omega** — razão entre ganhos e perdas acima de um limiar
- **Treynor** — excesso de retorno por unidade de risco sistemático (beta)
- **Jensen Alpha** — retorno acima do previsto pelo CAPM
- **Information Ratio / Tracking Error** — consistência do excesso de
  retorno em relação ao benchmark

Convenções: retornos simples por período (diários), taxa livre de risco
**anual**, resultados anualizados com 252 pregões.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from calahonda_var.metrics import (
    TRADING_DAYS,
    align_with_benchmark,
    annualized_return,
    annualized_volatility,
    beta_to_benchmark,
    max_drawdown,
)
from calahonda_var.var import sharpe_ratio


def _clean(returns) -> np.ndarray:
    r = np.asarray(returns, dtype=float).ravel()
    r = r[~np.isnan(r)]
    if r.size < 2:
        raise ValueError("`returns` precisa de pelo menos 2 observações válidas.")
    return r


def sortino_ratio(
    returns, risk_free: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> float:
    """Índice de Sortino anualizado.

    Igual ao Sharpe, mas o denominador é o *downside deviation*: a raiz
    da média dos quadrados apenas dos retornos abaixo da meta (aqui, a
    taxa livre de risco). Não pune volatilidade de alta.
    """
    r = _clean(returns)
    excess = r - risk_free / periods_per_year
    downside = np.minimum(excess, 0.0)
    downside_deviation = float(np.sqrt(np.mean(downside**2)))
    if downside_deviation == 0:
        raise ValueError("Sem retornos abaixo da meta: Sortino indefinido.")
    return float(excess.mean() / downside_deviation * np.sqrt(periods_per_year))


def calmar_ratio(returns, periods_per_year: int = TRADING_DAYS) -> float:
    """Índice de Calmar: retorno anualizado dividido pelo máximo drawdown."""
    drawdown = max_drawdown(returns)
    if drawdown == 0:
        raise ValueError("Máximo drawdown zero: Calmar indefinido.")
    return float(annualized_return(returns, periods_per_year) / drawdown)


def omega_ratio(returns, threshold: float = 0.0) -> float:
    """Índice Omega: soma dos ganhos ÷ soma das perdas acima do limiar.

    Usa a distribuição inteira (não só média e desvio). Omega > 1
    significa mais massa de ganhos que de perdas em torno do limiar
    (por período, ex.: diário).
    """
    r = _clean(returns)
    gains = float(np.clip(r - threshold, 0.0, None).sum())
    losses = float(np.clip(threshold - r, 0.0, None).sum())
    if losses == 0:
        raise ValueError("Sem perdas abaixo do limiar: Omega indefinido.")
    return gains / losses


def tracking_error(portfolio, benchmark, periods_per_year: int = TRADING_DAYS) -> float:
    """Tracking error anualizado: desvio do excesso de retorno diário."""
    aligned = align_with_benchmark(portfolio, benchmark)
    active = aligned["portfolio"] - aligned["benchmark"]
    deviation = float(active.std(ddof=1))
    return deviation * float(np.sqrt(periods_per_year))


def information_ratio(
    portfolio, benchmark, periods_per_year: int = TRADING_DAYS
) -> float:
    """Information Ratio: excesso de retorno anualizado ÷ tracking error."""
    aligned = align_with_benchmark(portfolio, benchmark)
    active = aligned["portfolio"] - aligned["benchmark"]
    error = tracking_error(portfolio, benchmark, periods_per_year)
    if error == 0:
        raise ValueError("Tracking error zero: Information Ratio indefinido.")
    return float(active.mean() * periods_per_year / error)


def treynor_ratio(
    portfolio,
    benchmark,
    risk_free: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Índice de Treynor: excesso de retorno anualizado ÷ beta."""
    beta = beta_to_benchmark(portfolio, benchmark)
    if beta == 0:
        raise ValueError("Beta zero: Treynor indefinido.")
    aligned = align_with_benchmark(portfolio, benchmark)
    excess = annualized_return(aligned["portfolio"], periods_per_year) - risk_free
    return float(excess / beta)


def jensen_alpha(
    portfolio,
    benchmark,
    risk_free: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Alfa de Jensen (CAPM), anualizado.

    Quanto a carteira rendeu **acima** do que o CAPM previa dado o seu
    beta: α = R_p − [rf + β·(R_b − rf)].
    """
    aligned = align_with_benchmark(portfolio, benchmark)
    beta = beta_to_benchmark(portfolio, benchmark)
    portfolio_return = annualized_return(aligned["portfolio"], periods_per_year)
    benchmark_return = annualized_return(aligned["benchmark"], periods_per_year)
    expected = risk_free + beta * (benchmark_return - risk_free)
    return float(portfolio_return - expected)


def metrics_report(portfolio, benchmark=None, risk_free: float = 0.0) -> pd.DataFrame:
    """Tabela legível com todas as métricas (para exibição).

    Métricas indefinidas para a amostra (ex.: Sortino sem perdas)
    aparecem como "—". Com ``benchmark``, inclui as métricas relativas
    (beta, Treynor, alfa, IR, tracking error).
    """

    def fmt(func, pattern: str) -> str:
        try:
            return pattern.format(func())
        except ValueError:
            return "—"

    rows: dict[str, str] = {
        "Retorno anualizado": fmt(lambda: annualized_return(portfolio), "{:.2%}"),
        "Volatilidade anual": fmt(lambda: annualized_volatility(portfolio), "{:.2%}"),
        "Máximo drawdown": fmt(lambda: max_drawdown(portfolio), "{:.2%}"),
        "Sharpe": fmt(lambda: sharpe_ratio(portfolio, risk_free), "{:.2f}"),
        "Sortino": fmt(lambda: sortino_ratio(portfolio, risk_free), "{:.2f}"),
        "Calmar": fmt(lambda: calmar_ratio(portfolio), "{:.2f}"),
        "Omega (limiar 0)": fmt(lambda: omega_ratio(portfolio), "{:.2f}"),
    }
    if benchmark is not None:
        rows |= {
            "Beta vs. benchmark": fmt(
                lambda: beta_to_benchmark(portfolio, benchmark), "{:.2f}"
            ),
            "Treynor": fmt(
                lambda: treynor_ratio(portfolio, benchmark, risk_free), "{:.2%}"
            ),
            "Jensen Alpha (a.a.)": fmt(
                lambda: jensen_alpha(portfolio, benchmark, risk_free), "{:.2%}"
            ),
            "Information Ratio": fmt(
                lambda: information_ratio(portfolio, benchmark), "{:.2f}"
            ),
            "Tracking error (a.a.)": fmt(
                lambda: tracking_error(portfolio, benchmark), "{:.2%}"
            ),
        }
    report = pd.DataFrame({"Valor": rows.values()}, index=pd.Index(rows.keys()))
    report.index.name = "Métrica"
    return report
