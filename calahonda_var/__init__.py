"""Calahonda VaR — Value at Risk e CVaR para carteiras da B3.

Três métodos clássicos da indústria (histórico, paramétrico e Monte
Carlo), mais Sharpe anualizado e carregamento de dados com fallback
sintético para rodar 100% offline.
"""

from calahonda_var.data import (
    DEFAULT_TICKERS,
    load_returns,
    portfolio_returns,
    synthetic_returns,
)
from calahonda_var.var import (
    RiskEstimate,
    historical_var,
    monte_carlo_var,
    parametric_var,
    sharpe_ratio,
    var_report,
)

__version__ = "1.0.0"

__all__ = [
    "DEFAULT_TICKERS",
    "RiskEstimate",
    "historical_var",
    "load_returns",
    "monte_carlo_var",
    "parametric_var",
    "portfolio_returns",
    "sharpe_ratio",
    "synthetic_returns",
    "var_report",
]
