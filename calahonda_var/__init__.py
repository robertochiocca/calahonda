"""Calahonda VaR — Value at Risk e CVaR para carteiras da B3.

Três métodos clássicos da indústria (histórico, paramétrico e Monte
Carlo), mais Sharpe anualizado e carregamento de dados com fallback
sintético para rodar 100% offline.
"""

from calahonda_var.backtest import VarBacktest, backtest_var, kupiec_test
from calahonda_var.data import (
    DEFAULT_TICKERS,
    load_benchmark,
    load_portfolio_csv,
    load_portfolio_file,
    load_returns,
    portfolio_returns,
    synthetic_returns,
)
from calahonda_var.metrics import (
    align_with_benchmark,
    annualized_return,
    annualized_volatility,
    beta_to_benchmark,
    correlation_matrix,
    drawdown_series,
    equity_curve,
    max_drawdown,
    max_sharpe_weights,
    min_variance_weights,
    portfolio_volatility,
)
from calahonda_var.report import generate_pdf_report
from calahonda_var.stress import (
    HISTORICAL_SCENARIOS,
    stress_test,
    volatility_shock_var,
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
    "HISTORICAL_SCENARIOS",
    "RiskEstimate",
    "VarBacktest",
    "align_with_benchmark",
    "annualized_return",
    "beta_to_benchmark",
    "backtest_var",
    "generate_pdf_report",
    "kupiec_test",
    "load_benchmark",
    "load_portfolio_csv",
    "load_portfolio_file",
    "stress_test",
    "volatility_shock_var",
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
