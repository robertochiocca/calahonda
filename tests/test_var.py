"""Testes do módulo de VaR/CVaR — 8 testes.

Inclui validação contra a teoria da distribuição normal: para retornos
normais de média zero, o VaR paramétrico a 95% deve ser ≈ 1.645·σ.

Uso::

    pytest                       # via pytest
    python tests/test_var.py     # execução direta
"""

import sys
from pathlib import Path

# Permite rodar os testes direto do clone, sem `pip install`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from scipy import stats

from calahonda_var import (
    historical_var,
    monte_carlo_var,
    parametric_var,
    portfolio_returns,
    sharpe_ratio,
    synthetic_returns,
    var_report,
)

RNG = np.random.default_rng(123)
NORMAL_SIGMA = 0.01
NORMAL_RETURNS = RNG.normal(0.0, NORMAL_SIGMA, size=100_000)


def test_historical_var_is_positive_and_cvar_dominates():
    """Para retornos com perdas, VaR > 0 e CVaR >= VaR (sempre)."""
    var, cvar = historical_var(NORMAL_RETURNS, confidence=0.95)
    assert var > 0
    assert cvar >= var


def test_parametric_var_matches_normal_theory():
    """Validação teórica: VaR 95% de uma normal μ=0 é 1.645·σ."""
    var, cvar = parametric_var(NORMAL_RETURNS, confidence=0.95)
    z = stats.norm.ppf(0.95)  # ≈ 1.6449
    expected_var = z * NORMAL_SIGMA
    expected_cvar = NORMAL_SIGMA * stats.norm.pdf(z) / 0.05
    assert var == pytest.approx(expected_var, rel=0.02)
    assert cvar == pytest.approx(expected_cvar, rel=0.02)


def test_monte_carlo_is_reproducible_with_seed():
    """Mesma semente => mesmo resultado; sementes diferentes => distinto."""
    a = monte_carlo_var(NORMAL_RETURNS, seed=42)
    b = monte_carlo_var(NORMAL_RETURNS, seed=42)
    c = monte_carlo_var(NORMAL_RETURNS, seed=99)
    assert a == b
    assert a.var != c.var


def test_monte_carlo_agrees_with_parametric_for_normal_data():
    """Com dados normais, Monte Carlo deve convergir ao paramétrico."""
    mc = monte_carlo_var(NORMAL_RETURNS, n_sims=200_000, seed=1)
    pa = parametric_var(NORMAL_RETURNS)
    assert mc.var == pytest.approx(pa.var, rel=0.03)
    assert mc.cvar == pytest.approx(pa.cvar, rel=0.03)


def test_higher_confidence_means_higher_var():
    """VaR 99% deve ser maior que VaR 95% em todos os métodos."""
    for method in (historical_var, parametric_var, monte_carlo_var):
        var95 = method(NORMAL_RETURNS, confidence=0.95).var
        var99 = method(NORMAL_RETURNS, confidence=0.99).var
        assert var99 > var95, method.__name__


def test_horizon_scales_with_square_root_of_time():
    """Com μ=0, VaR(21 dias) ≈ √21 · VaR(1 dia) no método paramétrico."""
    var_1d = parametric_var(NORMAL_RETURNS, horizon_days=1).var
    var_21d = parametric_var(NORMAL_RETURNS, horizon_days=21).var
    assert var_21d == pytest.approx(np.sqrt(21) * var_1d, rel=0.02)


def test_invalid_inputs_raise_value_error():
    """Parâmetros inválidos devem falhar cedo, com mensagem clara."""
    with pytest.raises(ValueError):
        historical_var(NORMAL_RETURNS, confidence=1.5)
    with pytest.raises(ValueError):
        parametric_var(NORMAL_RETURNS, horizon_days=0)
    with pytest.raises(ValueError):
        monte_carlo_var([0.01], confidence=0.95)  # poucos dados
    with pytest.raises(ValueError):
        monte_carlo_var(NORMAL_RETURNS, n_sims=10)
    with pytest.raises(ValueError):
        sharpe_ratio(np.zeros(100))  # desvio padrão zero
    with pytest.raises(ValueError):
        portfolio_returns(synthetic_returns(n_days=10), weights=[1.0])


def test_synthetic_data_and_report_are_consistent():
    """Dados sintéticos são reprodutíveis e o relatório cobre os 3 métodos."""
    a = synthetic_returns(n_days=252, seed=7)
    b = synthetic_returns(n_days=252, seed=7)
    assert a.shape == (252, 3)
    assert not a.isna().any().any()
    assert a.equals(b)

    portfolio = portfolio_returns(a)
    report = var_report(portfolio, portfolio_value=1_000_000.0)
    assert list(report.index) == ["historical", "parametric", "monte_carlo"]
    assert (report["CVaR %"] >= report["VaR %"]).all()
    assert (report["VaR R$"] > 0).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
