"""Stress testing: cenários históricos e choque de volatilidade.

Dois complementos ao VaR, que responde "quanto posso perder em condições
normais". O stress testing responde a pergunta oposta: "e se amanhã for
como 2008?".

- **Cenários históricos** — choques aproximados do Ibovespa em crises
  reais, aplicados ao valor da carteira (escaláveis por beta).
- **Choque de volatilidade** — recalcula o VaR paramétrico com a
  volatilidade multiplicada (ex.: 2×), simulando um regime de crise.
"""

from __future__ import annotations

import pandas as pd

from calahonda_var.var import RiskEstimate, parametric_var

# Quedas aproximadas do Ibovespa em episódios reais, para ordem de
# grandeza (fechamentos, topo a fundo no período indicado).
HISTORICAL_SCENARIOS: dict[str, float] = {
    "Joesley Day (mai/2017, 1 dia)": -0.088,
    "Semana dos circuit breakers (mar/2020)": -0.20,
    "COVID-19 (jan-mar/2020, topo a fundo)": -0.45,
    "Crise financeira global (2008, topo a fundo)": -0.60,
}


def stress_test(
    portfolio_value: float = 1_000_000.0,
    beta: float = 1.0,
    scenarios: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Perda da carteira em cada cenário histórico de crise.

    Parameters
    ----------
    portfolio_value : float
        Valor atual da carteira em R$.
    beta : float
        Sensibilidade da carteira ao índice (1.0 = cai igual ao
        Ibovespa; 0.5 = metade da queda). Deve ser positivo.
    scenarios : dict[str, float] | None
        Mapeamento nome → choque (fração negativa). ``None`` usa os
        cenários históricos padrão.

    Returns
    -------
    pandas.DataFrame
        Índice = cenário; colunas = Choque %, Perda R$, Valor final R$.
    """
    if portfolio_value <= 0:
        raise ValueError(
            f"`portfolio_value` deve ser positivo, recebeu {portfolio_value}."
        )
    if beta <= 0:
        raise ValueError(f"`beta` deve ser positivo, recebeu {beta}.")
    scenarios = HISTORICAL_SCENARIOS if scenarios is None else scenarios
    if not scenarios:
        raise ValueError("`scenarios` não pode ser vazio.")

    rows = {}
    for name, shock in scenarios.items():
        if shock >= 0:
            raise ValueError(
                f"Choque do cenário {name!r} deve ser negativo, recebeu {shock}."
            )
        effective = shock * beta
        loss = -effective * portfolio_value
        rows[name] = {
            "Choque %": round(effective * 100, 2),
            "Perda R$": round(loss, 2),
            "Valor final R$": round(portfolio_value - loss, 2),
        }
    report = pd.DataFrame.from_dict(rows, orient="index")
    report.index.name = "Cenário"
    return report


def volatility_shock_var(
    returns,
    vol_multiplier: float = 2.0,
    confidence: float = 0.95,
    horizon_days: int = 1,
) -> RiskEstimate:
    """VaR paramétrico sob um regime de volatilidade multiplicada.

    Escala os retornos pelo multiplicador (preservando a média em zero
    relativo), o que equivale a recalcular o VaR com σ' = m·σ.
    """
    if vol_multiplier <= 0:
        raise ValueError(
            f"`vol_multiplier` deve ser positivo, recebeu {vol_multiplier}."
        )
    scaled = pd.Series(returns).dropna() * vol_multiplier
    return parametric_var(scaled, confidence=confidence, horizon_days=horizon_days)
