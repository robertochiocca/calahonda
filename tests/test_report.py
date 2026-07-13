"""Teste do relatório em PDF — 1 teste (requer matplotlib).

Uso::

    pytest                      # via pytest
    python tests/test_report.py # execução direta
"""

import io
import sys
from pathlib import Path

# Permite rodar os testes direto do clone, sem `pip install`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

pytest.importorskip("matplotlib", reason="relatório em PDF requer matplotlib")

from calahonda_var import generate_pdf_report, portfolio_returns, synthetic_returns


def test_pdf_report_has_three_pages_in_memory():
    """O relatório é um PDF válido, com 3 páginas, gerado em memória."""
    portfolio = portfolio_returns(synthetic_returns(seed=7))
    buffer = io.BytesIO()
    result = generate_pdf_report(portfolio, buffer, portfolio_value=1_000_000.0)
    assert result is buffer

    raw = buffer.getvalue()
    assert raw.startswith(b"%PDF")
    n_pages = raw.count(b"/Type /Page") - raw.count(b"/Type /Pages")
    assert n_pages == 3
    assert len(raw) > 10_000  # tem conteúdo real, não um PDF vazio


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
