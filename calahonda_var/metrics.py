"""Métricas de performance e otimização de carteira.

Complementa o módulo de VaR com as métricas que gestores acompanham no
dia a dia — retorno e volatilidade anualizados, curva de patrimônio,
drawdown e correlação — e com otimização de carteira (mínima variância
e máximo Sharpe) sob restrições realistas: sem alavancagem e sem
posições vendidas (0 ≤ peso ≤ 1, soma = 1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252


def _clean_series(returns) -> np.ndarray:
    r = np.asarray(returns, dtype=float).ravel()
    r = r[~np.isnan(r)]
    if r.size < 2:
        raise ValueError("`returns` precisa de pelo menos 2 observações válidas.")
    return r


def _clean_frame(returns: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("`returns` deve ser um pandas.DataFrame (um ativo por coluna).")
    clean = returns.dropna(how="any")
    if clean.shape[0] < 2 or clean.shape[1] < 2:
        raise ValueError(
            "`returns` precisa de pelo menos 2 observações e 2 ativos, "
            f"recebeu {clean.shape}."
        )
    return clean


def annualized_return(returns, periods_per_year: int = TRADING_DAYS) -> float:
    """Retorno anualizado (média aritmética × períodos por ano).

    Aproximação padrão para métricas de risco de curto prazo; para
    performance composta de longo prazo, prefira o retorno geométrico.
    """
    r = _clean_series(returns)
    return float(r.mean() * periods_per_year)


def annualized_volatility(returns, periods_per_year: int = TRADING_DAYS) -> float:
    """Volatilidade anualizada (desvio padrão × √períodos por ano)."""
    r = _clean_series(returns)
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def equity_curve(returns, initial: float = 1.0) -> pd.Series:
    """Curva de patrimônio: capital acumulado a partir dos retornos.

    Parameters
    ----------
    initial : float
        Capital inicial (1.0 = base 100%).
    """
    r = pd.Series(returns).dropna()
    if r.size < 2:
        raise ValueError("`returns` precisa de pelo menos 2 observações válidas.")
    curve = (1.0 + r).cumprod() * initial
    curve.name = "equity"
    return curve


def drawdown_series(returns) -> pd.Series:
    """Série de drawdown: queda percentual desde o pico anterior (valores ≤ 0)."""
    curve = equity_curve(returns)
    dd = curve / curve.cummax() - 1.0
    dd.name = "drawdown"
    return dd


def max_drawdown(returns) -> float:
    """Máximo drawdown como fração positiva (0.25 = queda de 25% do pico)."""
    return float(-drawdown_series(returns).min())


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Matriz de correlação entre os ativos (Pearson)."""
    return _clean_frame(returns).corr()


def portfolio_volatility(
    returns: pd.DataFrame, weights, periods_per_year: int = TRADING_DAYS
) -> float:
    """Volatilidade anualizada da carteira: √(wᵀ·Σ·w) × √períodos."""
    clean = _clean_frame(returns)
    w = np.asarray(weights, dtype=float).ravel()
    if w.size != clean.shape[1]:
        raise ValueError(f"`weights` tem {w.size} pesos para {clean.shape[1]} ativos.")
    cov = clean.cov().to_numpy()
    return float(np.sqrt(w @ cov @ w) * np.sqrt(periods_per_year))


def _optimize(returns: pd.DataFrame, objective) -> pd.Series:
    """Resolve a otimização long-only (0 ≤ w ≤ 1, Σw = 1) via SLSQP."""
    clean = _clean_frame(returns)
    n = clean.shape[1]
    result = minimize(
        objective,
        x0=np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
    )
    if not result.success:
        raise RuntimeError(f"Otimização não convergiu: {result.message}")
    weights = np.clip(result.x, 0.0, 1.0)
    weights = weights / weights.sum()
    return pd.Series(weights, index=clean.columns, name="weight")


def min_variance_weights(returns: pd.DataFrame) -> pd.Series:
    """Pesos da carteira de mínima variância (long-only, sem alavancagem).

    Returns
    -------
    pandas.Series
        Peso por ativo, somando 1.
    """
    # Variância anualizada: escala o objetivo para longe da tolerância
    # numérica do SLSQP (a variância diária é ~1e-4).
    cov = _clean_frame(returns).cov().to_numpy() * TRADING_DAYS
    return _optimize(returns, lambda w: w @ cov @ w)


def max_sharpe_weights(returns: pd.DataFrame, risk_free: float = 0.0) -> pd.Series:
    """Pesos da carteira de máximo Sharpe (long-only, sem alavancagem).

    Parameters
    ----------
    risk_free : float
        Taxa livre de risco **anual** (ex.: 0.10 para 10% a.a.).
    """
    clean = _clean_frame(returns)
    mu = clean.mean().to_numpy()
    cov = clean.cov().to_numpy()
    rf_daily = risk_free / TRADING_DAYS

    def negative_sharpe(w):
        vol = np.sqrt(w @ cov @ w)
        if vol == 0:
            return 0.0
        return -(w @ mu - rf_daily) / vol

    return _optimize(returns, negative_sharpe)
