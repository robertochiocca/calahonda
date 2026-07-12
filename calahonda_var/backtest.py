"""Backtesting de VaR: violações observadas × esperadas e teste de Kupiec.

Um modelo de VaR 95% "aprovado" deve ser violado em ~5% dos dias — nem
muito mais (subestima o risco) nem muito menos (superestima e trava
capital). O teste de Kupiec (proportion-of-failures, 1995) verifica
estatisticamente se a frequência observada de violações é compatível com
a esperada.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import stats


class VarBacktest(NamedTuple):
    """Resultado do backtest de VaR."""

    observations: int
    violations: int
    expected_violations: float
    violation_rate: float
    kupiec_statistic: float
    kupiec_pvalue: float
    approved: bool


def kupiec_test(violations: int, observations: int, confidence: float = 0.95):
    """Teste de Kupiec (POF): estatística LR e p-valor (χ², 1 g.l.).

    H0: a taxa de violações observada é compatível com ``1 − confidence``.
    p-valor < 0.05 rejeita o modelo de VaR.
    """
    if observations <= 0:
        raise ValueError(f"`observations` deve ser positivo, recebeu {observations}.")
    if not 0 <= violations <= observations:
        raise ValueError("`violations` deve estar entre 0 e `observations`.")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"`confidence` deve estar em (0, 1), recebeu {confidence}.")

    p = 1.0 - confidence
    x, n = violations, observations
    rate = x / n

    # log-verossimilhança sob H0 e sob a taxa observada (com guardas
    # para x = 0 e x = n, onde 0·log(0) := 0)
    def _ll(prob: float) -> float:
        with np.errstate(divide="ignore", invalid="ignore"):
            term_hit = x * np.log(prob) if x > 0 else 0.0
            term_miss = (n - x) * np.log(1.0 - prob) if x < n else 0.0
        return term_hit + term_miss

    lr = -2.0 * (_ll(p) - _ll(rate))
    pvalue = float(1.0 - stats.chi2.cdf(lr, df=1))
    return float(lr), pvalue


def backtest_var(
    returns,
    confidence: float = 0.95,
    window: int = 252,
) -> VarBacktest:
    """Backtest de VaR histórico com janela móvel (out-of-sample).

    Para cada dia t, estima o VaR de 1 dia com os ``window`` retornos
    anteriores e verifica se a perda realizada em t excedeu a estimativa.
    Ao final, aplica o teste de Kupiec sobre a taxa de violações.

    Parameters
    ----------
    returns : array-like
        Retornos diários simples.
    confidence : float
        Nível de confiança do VaR em (0, 1).
    window : int
        Janela de estimação em dias úteis (>= 60).
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"`confidence` deve estar em (0, 1), recebeu {confidence}.")
    if window < 60:
        raise ValueError(f"`window` deve ser >= 60, recebeu {window}.")

    losses = pd.Series(np.asarray(returns, dtype=float).ravel()).dropna().mul(-1.0)
    if len(losses) < window + 30:
        raise ValueError(
            f"Backtest precisa de pelo menos window+30 = {window + 30} "
            f"observações, recebeu {len(losses)}."
        )

    # VaR estimado com dados ATÉ ontem (shift(1)) — nada de olhar o futuro
    var_estimates = losses.rolling(window).quantile(confidence).shift(1)
    valid = var_estimates.notna()
    hits = losses[valid] > var_estimates[valid]

    n = int(valid.sum())
    x = int(hits.sum())
    lr, pvalue = kupiec_test(x, n, confidence)

    return VarBacktest(
        observations=n,
        violations=x,
        expected_violations=round(n * (1.0 - confidence), 1),
        violation_rate=round(x / n, 4),
        kupiec_statistic=round(lr, 4),
        kupiec_pvalue=round(pvalue, 4),
        approved=pvalue > 0.05,
    )
