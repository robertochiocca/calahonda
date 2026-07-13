"""Testes das métricas avançadas e do cenário personalizado — 4 testes.

Uso::

    pytest                      # via pytest
    python tests/test_ratios.py # execução direta
"""

import sys
from pathlib import Path

# Permite rodar os testes direto do clone, sem `pip install`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from calahonda_var import (
    annualized_return,
    calmar_ratio,
    information_ratio,
    jensen_alpha,
    max_drawdown,
    metrics_report,
    omega_ratio,
    sortino_ratio,
    tracking_error,
    treynor_ratio,
    weighted_shock,
)

RNG = np.random.default_rng(9)
RETURNS = pd.Series(RNG.normal(0.0005, 0.012, 1_000))


def test_sortino_calmar_omega_match_definitions():
    """Cada índice bate com a fórmula calculada à mão."""
    r = RETURNS.to_numpy()
    downside = np.minimum(r, 0.0)
    expected_sortino = r.mean() / np.sqrt((downside**2).mean()) * np.sqrt(252)
    assert sortino_ratio(RETURNS) == pytest.approx(expected_sortino)

    assert calmar_ratio(RETURNS) == pytest.approx(
        annualized_return(RETURNS) / max_drawdown(RETURNS)
    )

    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    assert omega_ratio(RETURNS) == pytest.approx(gains / losses)

    with pytest.raises(ValueError):
        sortino_ratio(np.full(100, 0.01))  # sem perdas: indefinido
    with pytest.raises(ValueError):
        omega_ratio(np.full(100, 0.01))


def test_benchmark_relative_ratios_known_cases():
    """Contra si mesmo: alfa 0, tracking error 0, Treynor = retorno."""
    assert jensen_alpha(RETURNS, RETURNS) == pytest.approx(0.0, abs=1e-12)
    assert tracking_error(RETURNS, RETURNS) == pytest.approx(0.0, abs=1e-12)
    assert treynor_ratio(RETURNS, RETURNS) == pytest.approx(annualized_return(RETURNS))
    with pytest.raises(ValueError):
        information_ratio(RETURNS, RETURNS)  # tracking error zero

    # beta 2 => Treynor cai à metade do excesso de retorno da carteira
    doubled = RETURNS * 2
    assert treynor_ratio(doubled, RETURNS) == pytest.approx(
        annualized_return(doubled) / 2
    )


def test_metrics_report_structure():
    """Sem benchmark: 7 linhas; com benchmark: +5 relativas; erros viram '—'."""
    solo = metrics_report(RETURNS)
    assert len(solo) == 7
    assert "Sharpe" in solo.index

    full = metrics_report(RETURNS, benchmark=RETURNS)
    assert len(full) == 12
    assert full.loc["Information Ratio", "Valor"] == "—"  # TE zero → indefinido
    assert full.loc["Beta vs. benchmark", "Valor"] == "1.00"


def test_weighted_shock_custom_scenario():
    """Cenário personalizado: Σ peso·choque, com normalização e validação."""
    # Petrobras -15%, Vale -8%, Itaú +2% com pesos 40/30/30
    shock = weighted_shock([40, 30, 30], [-0.15, -0.08, 0.02])
    assert shock == pytest.approx(0.4 * -0.15 + 0.3 * -0.08 + 0.3 * 0.02)

    with pytest.raises(ValueError):
        weighted_shock([1, 1], [-0.1])  # tamanhos diferentes
    with pytest.raises(ValueError):
        weighted_shock([], [])
    with pytest.raises(ValueError):
        weighted_shock([0, 0], [-0.1, -0.1])  # pesos somam zero


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
