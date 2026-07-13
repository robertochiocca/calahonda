"""Exemplo completo: VaR de uma carteira de R$1M na B3.

Carteira com pesos iguais em PETR4 + VALE3 + ITUB4, VaR 95% em um
horizonte de 21 dias úteis, pelos três métodos.

Uso::

    python examples/example_var.py

Funciona offline (dados sintéticos). Com ``yfinance`` instalado e
internet disponível, usa preços reais da B3 automaticamente.
"""

import sys
from pathlib import Path

# Permite rodar o exemplo direto do clone, sem `pip install`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from calahonda_var import (
    load_returns,
    portfolio_returns,
    sharpe_ratio,
    var_report,
)

PORTFOLIO_VALUE = 1_000_000.0  # R$
CONFIDENCE = 0.95
HORIZON_DAYS = 21
TRADING_DAYS = 252


def main() -> None:
    returns = load_returns()
    portfolio = portfolio_returns(returns)

    print("=" * 62)
    print("Calahonda VaR — carteira de R$1M em PETR4 + VALE3 + ITUB4")
    print(f"VaR {CONFIDENCE:.0%} · horizonte de {HORIZON_DAYS} dias úteis")
    print(
        f"Amostra: {len(portfolio)} pregões "
        f"({portfolio.index[0]:%Y-%m-%d} a {portfolio.index[-1]:%Y-%m-%d})"
    )
    print("=" * 62)
    print()

    report = var_report(
        portfolio,
        portfolio_value=PORTFOLIO_VALUE,
        confidence=CONFIDENCE,
        horizon_days=HORIZON_DAYS,
    )
    print(report)
    print()

    ann_return = portfolio.mean() * TRADING_DAYS
    ann_vol = portfolio.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = sharpe_ratio(portfolio)

    print(f"Retorno anualizado : {ann_return:>7.2%}")
    print(f"Volatilidade anual : {ann_vol:>7.2%}")
    print(f"Sharpe (rf=0)      : {sharpe:>7.2f}")


if __name__ == "__main__":
    main()
