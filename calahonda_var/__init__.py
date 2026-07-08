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
from calahonda_var.metrics import (
    annualized_return,
    annualized_volatility,
    correlation_matrix,
    drawdown_series,
    equity_curve,
    max_drawdown,
    max_sharpe_weights,
    min_variance_weights,
    portfolio_volatility,
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
    "annualized_return",
    "annualized_volatility",
    "correlation_matrix",
    "drawdown_series",
    "equity_curve",
    "historical_var",
    "load_returns",
    "max_drawdown",
    "max_sharpe_weights",
    "min_variance_weights",
    "monte_carlo_var",
    "parametric_var",
    "portfolio_returns",
    "portfolio_volatility",
    "sharpe_ratio",
    "synthetic_returns",
    "var_report",
]
