"""Testes das métricas de performance e otimização — 6 testes.

Uso::

    pytest                          # via pytest
    python tests/test_metrics.py    # execução direta
"""

import sys
from pathlib import Path

# Permite rodar os testes direto do clone, sem `pip install`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from calahonda_var import (
    annualized_return,
    annualized_volatility,
    correlation_matrix,
    drawdown_series,
    equity_curve,
    max_drawdown,
    max_sharpe_weights,
    min_variance_weights,
    portfolio_volatility,
    sharpe_ratio,
    synthetic_returns,
)

RETURNS = synthetic_returns(n_days=756, seed=7)


def test_annualized_metrics_match_definitions():
    """Anualização segue as definições: μ·252 e σ·√252."""
    r = RETURNS.iloc[:, 0]
    assert annualized_return(r) == pytest.approx(r.mean() * 252)
    assert annualized_volatility(r) == pytest.approx(r.std(ddof=1) * np.sqrt(252))


def test_max_drawdown_known_case():
    """Caso construído à mão: +10%, −50%, +20% → drawdown máximo de 50%."""
    returns = [0.10, -0.50, 0.20]
    curve = equity_curve(returns)
    assert curve.iloc[-1] == pytest.approx(1.10 * 0.50 * 1.20)
    assert max_drawdown(returns) == pytest.approx(0.50)
    dd = drawdown_series(returns)
    assert (dd <= 0).all()
    assert dd.min() == pytest.approx(-0.50)


def test_correlation_matrix_properties():
    """Diagonal 1, simétrica e valores dentro de [-1, 1]."""
    corr = correlation_matrix(RETURNS)
    assert np.allclose(np.diag(corr), 1.0)
    assert np.allclose(corr, corr.T)
    assert (corr.abs() <= 1.0 + 1e-12).all().all()


def test_min_variance_beats_equal_weight():
    """A carteira de mínima variância não pode ser mais volátil que a 1/N."""
    weights = min_variance_weights(RETURNS)
    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0).all()
    n = RETURNS.shape[1]
    vol_opt = portfolio_volatility(RETURNS, weights)
    vol_eq = portfolio_volatility(RETURNS, np.full(n, 1 / n))
    assert vol_opt <= vol_eq + 1e-9


def test_max_sharpe_beats_equal_weight():
    """A carteira de máximo Sharpe não pode perder para a 1/N em Sharpe."""
    weights = max_sharpe_weights(RETURNS)
    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0).all()
    sharpe_opt = sharpe_ratio(RETURNS @ weights.to_numpy())
    n = RETURNS.shape[1]
    sharpe_eq = sharpe_ratio(RETURNS @ np.full(n, 1 / n))
    assert sharpe_opt >= sharpe_eq - 1e-9


def test_metrics_invalid_inputs_raise():
    """Entradas inválidas falham cedo, com mensagem clara."""
    with pytest.raises(ValueError):
        annualized_return([0.01])  # poucos dados
    with pytest.raises(TypeError):
        correlation_matrix([0.01, 0.02])  # não é DataFrame
    with pytest.raises(ValueError):
        portfolio_volatility(RETURNS, [0.5, 0.5])  # nº de pesos errado


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
