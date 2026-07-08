"""Cálculo de Value at Risk (VaR) e Conditional VaR (CVaR).

Três métodos clássicos da indústria:

- **Histórico**   — quantil da distribuição empírica dos retornos
- **Paramétrico** — variância-covariância (aproximação normal)
- **Monte Carlo** — simulação de cenários i.i.d. normais

Convenção: VaR e CVaR são retornados como números **positivos** em
fração do valor da carteira (``var = 0.13`` significa perda de 13%).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import stats

ArrayLike = "np.ndarray | pd.Series | list[float]"


class RiskEstimate(NamedTuple):
    """Par (VaR, CVaR) em fração do valor da carteira."""

    var: float
    cvar: float


def _clean_returns(
    returns, confidence: float, horizon_days: int
) -> np.ndarray:
    """Valida os parâmetros e devolve os retornos como array 1-D sem NaN."""
    r = np.asarray(returns, dtype=float).ravel()
    r = r[~np.isnan(r)]
    if r.size < 2:
        raise ValueError(
            "`returns` precisa de pelo menos 2 observações válidas, "
            f"recebeu {r.size}."
        )
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"`confidence` deve estar em (0, 1), recebeu {confidence}.")
    if horizon_days < 1:
        raise ValueError(f"`horizon_days` deve ser >= 1, recebeu {horizon_days}.")
    return r


def _cvar_from_losses(losses: np.ndarray, var: float) -> float:
    """CVaR empírico: média das perdas iguais ou além do VaR."""
    tail = losses[losses >= var]
    # Com poucos pontos o quantil pode ficar acima da maior perda simulada;
    # nesse caso o CVaR colapsa para o próprio VaR.
    return float(tail.mean()) if tail.size else float(var)


def historical_var(
    returns, confidence: float = 0.95, horizon_days: int = 1
) -> RiskEstimate:
    """VaR e CVaR pelo método histórico (distribuição empírica).

    O quantil é extraído diretamente das perdas observadas em 1 dia e
    escalado para o horizonte pela regra da raiz do tempo (√t).

    Parameters
    ----------
    returns : array-like
        Retornos simples por período (ex.: diários).
    confidence : float
        Nível de confiança em (0, 1). Padrão 0.95.
    horizon_days : int
        Horizonte em dias úteis. Padrão 1.
    """
    r = _clean_returns(returns, confidence, horizon_days)
    losses = -r
    var_1d = float(np.quantile(losses, confidence))
    cvar_1d = _cvar_from_losses(losses, var_1d)
    scale = float(np.sqrt(horizon_days))
    return RiskEstimate(var_1d * scale, cvar_1d * scale)


def parametric_var(
    returns, confidence: float = 0.95, horizon_days: int = 1
) -> RiskEstimate:
    """VaR e CVaR paramétricos (variância-covariância, normal).

    Fórmulas fechadas sob normalidade, com ``z = Φ⁻¹(1 − confidence)``::

        VaR  = −(μ·h + z·σ·√h)
        CVaR = −(μ·h − σ·√h · φ(z) / (1 − confidence))
    """
    r = _clean_returns(returns, confidence, horizon_days)
    mu = r.mean() * horizon_days
    sigma = r.std(ddof=1) * np.sqrt(horizon_days)
    alpha = 1.0 - confidence
    z = stats.norm.ppf(alpha)
    var = -(mu + z * sigma)
    cvar = -(mu - sigma * stats.norm.pdf(z) / alpha)
    return RiskEstimate(float(var), float(cvar))


def monte_carlo_var(
    returns,
    confidence: float = 0.95,
    horizon_days: int = 1,
    n_sims: int = 10_000,
    seed: int | None = 42,
) -> RiskEstimate:
    """VaR e CVaR por simulação de Monte Carlo.

    Estima média e desvio dos retornos, simula ``n_sims`` trajetórias
    i.i.d. normais ao longo do horizonte e extrai o quantil das perdas
    acumuladas.

    Parameters
    ----------
    n_sims : int
        Número de cenários simulados. Padrão 10.000.
    seed : int | None
        Semente do gerador para resultados reprodutíveis. Use ``None``
        para uma simulação diferente a cada chamada.
    """
    r = _clean_returns(returns, confidence, horizon_days)
    if n_sims < 100:
        raise ValueError(f"`n_sims` deve ser >= 100, recebeu {n_sims}.")

    mu = r.mean()
    sigma = r.std(ddof=1)
    rng = np.random.default_rng(seed)

    sims = rng.normal(mu, sigma, size=(n_sims, horizon_days))
    losses = -sims.sum(axis=1)

    var = float(np.quantile(losses, confidence))
    cvar = _cvar_from_losses(losses, var)
    return RiskEstimate(var, cvar)


def sharpe_ratio(
    returns, risk_free: float = 0.0, periods_per_year: int = 252
) -> float:
    """Índice de Sharpe anualizado.

    Parameters
    ----------
    returns : array-like
        Retornos simples por período.
    risk_free : float
        Taxa livre de risco **anual** (ex.: 0.10 para 10% a.a.).
    periods_per_year : int
        Períodos por ano (252 para retornos diários).
    """
    r = np.asarray(returns, dtype=float).ravel()
    r = r[~np.isnan(r)]
    if r.size < 2:
        raise ValueError("`returns` precisa de pelo menos 2 observações válidas.")
    excess = r - risk_free / periods_per_year
    sd = excess.std(ddof=1)
    if sd == 0:
        raise ValueError("Desvio padrão zero: Sharpe indefinido.")
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def var_report(
    returns,
    portfolio_value: float = 1_000_000.0,
    confidence: float = 0.95,
    horizon_days: int = 1,
    n_sims: int = 10_000,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Compara os três métodos de VaR em uma única tabela.

    Returns
    -------
    pandas.DataFrame
        Índice = método; colunas = VaR %, CVaR %, VaR R$, CVaR R$.
    """
    estimates = {
        "historical": historical_var(returns, confidence, horizon_days),
        "parametric": parametric_var(returns, confidence, horizon_days),
        "monte_carlo": monte_carlo_var(returns, confidence, horizon_days, n_sims, seed),
    }
    report = pd.DataFrame(
        {
            "VaR %": [e.var * 100 for e in estimates.values()],
            "CVaR %": [e.cvar * 100 for e in estimates.values()],
            "VaR R$": [e.var * portfolio_value for e in estimates.values()],
            "CVaR R$": [e.cvar * portfolio_value for e in estimates.values()],
        },
        index=pd.Index(estimates.keys(), name="Método"),
    )
    return report.round({"VaR %": 3, "CVaR %": 3, "VaR R$": 2, "CVaR R$": 2})
