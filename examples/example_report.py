"""Gera o relatório de risco em PDF a partir de um CSV de carteira.

Uso::

    pip install matplotlib
    python examples/example_report.py

Lê ``examples/carteira_exemplo.csv``, monta a carteira e salva o PDF em
``docs/relatorio_exemplo.pdf``. Funciona offline (dados sintéticos).
"""

import sys
from pathlib import Path

# Permite rodar o exemplo direto do clone, sem `pip install`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calahonda_var import (
    generate_pdf_report,
    load_portfolio_csv,
    load_returns,
    portfolio_returns,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    tickers, weights = load_portfolio_csv(ROOT / "examples" / "carteira_exemplo.csv")
    returns = load_returns(tickers)
    portfolio = portfolio_returns(returns, weights)

    output = ROOT / "docs" / "relatorio_exemplo.pdf"
    output.parent.mkdir(exist_ok=True)
    generate_pdf_report(portfolio, output, portfolio_value=1_000_000.0)
    print(f"Relatório salvo em {output}")


if __name__ == "__main__":
    main()
