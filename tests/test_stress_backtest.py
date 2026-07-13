"""Testes de stress testing e backtesting de VaR — 4 testes.

Uso::

    pytest                               # via pytest
    python tests/test_stress_backtest.py # execução direta
"""

import sys
from pathlib import Path

# Permite rodar os testes direto do clone, sem `pip install`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import io

import numpy as np
import pytest

from calahonda_var import (
    HISTORICAL_SCENARIOS,
    backtest_var,
    kupiec_test,
    load_portfolio_csv,
    stress_test,
    volatility_shock_var,
)


def test_stress_test_scenarios_and_arithmetic():
    """Todos os cenários aparecem e a aritmética das perdas fecha."""
    value = 1_000_000.0
    report = stress_test(portfolio_value=value)
    assert len(report) == len(HISTORICAL_SCENARIOS)
    assert (report["Perda R$"] > 0).all()
    worst = HISTORICAL_SCENARIOS["Crise financeira global (2008, topo a fundo)"]
    row = report.loc["Crise financeira global (2008, topo a fundo)"]
    assert row["Perda R$"] == pytest.approx(-worst * value)
    assert row["Valor final R$"] == pytest.approx(value * (1 + worst))
    # beta = 0.5 => metade da perda
    half = stress_test(portfolio_value=value, beta=0.5)
    assert half["Perda R$"].iloc[0] == pytest.approx(report["Perda R$"].iloc[0] / 2)
    # choque de vol 2x ~ dobra o VaR paramétrico de média ~0
    rng = np.random.default_rng(5)
    r = rng.normal(0.0, 0.01, 20_000)
    assert volatility_shock_var(r, 2.0).var == pytest.approx(
        2 * volatility_shock_var(r, 1.0).var, rel=0.01
    )


def test_backtest_approves_wellspecified_var():
    """Em dados i.i.d. normais, o VaR histórico deve ser aprovado (~5%)."""
    rng = np.random.default_rng(11)
    r = rng.normal(0.0004, 0.015, 2_000)
    result = backtest_var(r, confidence=0.95, window=252)
    assert result.observations == 2_000 - 252
    assert 0.03 < result.violation_rate < 0.07
    assert result.approved


def test_kupiec_rejects_bad_var_model():
    """Kupiec: 12/250 violações a 95% passa; 50/250 é rejeitado."""
    _, p_ok = kupiec_test(12, 250, confidence=0.95)
    _, p_bad = kupiec_test(50, 250, confidence=0.95)
    assert p_ok > 0.05
    assert p_bad < 0.01
    # casos extremos não podem explodir (0·log 0)
    lr_zero, _ = kupiec_test(0, 250, confidence=0.95)
    assert np.isfinite(lr_zero)


def test_csv_import_and_invalid_inputs():
    """CSV de carteira normaliza pesos e adiciona .SA; inválidos falham."""
    csv = io.StringIO("ticker,peso\nPETR4,40\nVALE3,30\nITUB4,30\n")
    tickers, weights = load_portfolio_csv(csv)
    assert tickers == ("PETR4.SA", "VALE3.SA", "ITUB4.SA")
    assert weights.sum() == pytest.approx(1.0)
    assert weights[0] == pytest.approx(0.4)

    with pytest.raises(ValueError):
        load_portfolio_csv(io.StringIO("acao,quantidade\nPETR4,10\n"))
    with pytest.raises(ValueError):
        stress_test(portfolio_value=-1)
    with pytest.raises(ValueError):
        stress_test(beta=0)
    with pytest.raises(ValueError):
        backtest_var(np.zeros(100), window=252)  # poucos dados
    with pytest.raises(ValueError):
        kupiec_test(10, 0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
