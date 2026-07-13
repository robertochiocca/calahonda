"""Testes de importação multi-formato e benchmark — 2 testes.

Uso::

    pytest                              # via pytest
    python tests/test_data_benchmark.py # execução direta
"""

import io
import sys
from pathlib import Path

# Permite rodar os testes direto do clone, sem `pip install`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from calahonda_var import (
    align_with_benchmark,
    beta_to_benchmark,
    load_benchmark,
    load_portfolio_file,
    portfolio_returns,
    synthetic_returns,
)


def test_load_portfolio_file_xlsx_and_json(tmp_path):
    """Excel e JSON produzem a mesma carteira que o CSV equivalente."""
    frame = pd.DataFrame({"ticker": ["PETR4", "VALE3", "ITUB4"], "peso": [40, 30, 30]})

    xlsx_path = tmp_path / "carteira.xlsx"
    frame.to_excel(xlsx_path, index=False)
    tickers_xlsx, weights_xlsx = load_portfolio_file(xlsx_path)

    json_buffer = io.StringIO(frame.to_json(orient="records"))
    tickers_json, weights_json = load_portfolio_file(json_buffer, name="carteira.json")

    expected = ("PETR4.SA", "VALE3.SA", "ITUB4.SA")
    assert tickers_xlsx == expected
    assert tickers_json == expected
    assert weights_xlsx == pytest.approx([0.4, 0.3, 0.3])
    assert weights_json == pytest.approx(weights_xlsx)

    with pytest.raises(ValueError):
        load_portfolio_file(io.StringIO("x"), name="carteira.txt")


def test_benchmark_alignment_and_beta():
    """Alinhamento por datas em comum e beta com valores conhecidos."""
    benchmark = load_benchmark(fallback=True)  # offline: sintético
    assert isinstance(benchmark, pd.Series)
    assert len(benchmark) > 100

    portfolio = portfolio_returns(synthetic_returns())
    aligned = align_with_benchmark(portfolio, benchmark)
    assert list(aligned.columns) == ["portfolio", "benchmark"]
    assert len(aligned) > 100

    # casos exatos: beta(x, x) = 1 e beta(2x, x) = 2
    x = pd.Series(np.random.default_rng(3).normal(0, 0.01, 500))
    assert beta_to_benchmark(x, x) == pytest.approx(1.0)
    assert beta_to_benchmark(2 * x, x) == pytest.approx(2.0)

    with pytest.raises(ValueError):
        beta_to_benchmark(x, pd.Series(np.zeros(500)))  # variância zero


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
