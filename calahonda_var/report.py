"""Relatório da carteira em PDF (matplotlib, sem dependências extras).

Gera um PDF de 3 páginas: resumo com métricas e stress testing, curva de
patrimônio com drawdown, e distribuição de Monte Carlo. Requer
``matplotlib`` (extra opcional ``viz``).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from calahonda_var.backtest import backtest_var
from calahonda_var.metrics import (
    annualized_return,
    annualized_volatility,
    drawdown_series,
    equity_curve,
    max_drawdown,
)
from calahonda_var.stress import stress_test
from calahonda_var.theme import AMBER as _AMBER
from calahonda_var.theme import INK as _INK
from calahonda_var.theme import LIME as _LIME
from calahonda_var.theme import MATPLOTLIB_RC
from calahonda_var.theme import MUTED as _MUTED
from calahonda_var.theme import RED as _RED
from calahonda_var.theme import SURFACE as _SURFACE
from calahonda_var.var import monte_carlo_var, sharpe_ratio, var_report


def generate_pdf_report(
    portfolio: pd.Series,
    output,
    portfolio_value: float = 1_000_000.0,
    confidence: float = 0.95,
    horizon_days: int = 21,
    title: str = "Relatório de Risco — Calahonda",
):
    """Gera o relatório em PDF e devolve o destino (caminho ou buffer).

    Parameters
    ----------
    portfolio : pandas.Series
        Retornos diários da carteira (use ``portfolio_returns``).
    output : str | Path | file-like
        Caminho do PDF ou buffer (ex.: ``io.BytesIO`` para download).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    plt.rcParams.update({**MATPLOTLIB_RC, "font.size": 9})

    report = var_report(portfolio, portfolio_value, confidence, horizon_days)
    bt = backtest_var(portfolio, confidence=confidence)
    stress = stress_test(portfolio_value)

    with PdfPages(output) as pdf:
        # ---- Página 1: resumo executivo ----
        fig = plt.figure(figsize=(8.27, 11.69))  # A4
        fig.text(0.07, 0.95, title, fontsize=18, fontweight="bold", color=_LIME)
        fig.text(
            0.07,
            0.925,
            f"Gerado em {date.today():%d/%m/%Y} · "
            f"carteira de R$ {portfolio_value:,.0f} · VaR {confidence:.0%} · "
            f"{horizon_days} dias úteis",
            fontsize=9,
            color=_MUTED,
        )

        lines = [
            ("Retorno anualizado", f"{annualized_return(portfolio):.2%}"),
            ("Volatilidade anual", f"{annualized_volatility(portfolio):.2%}"),
            ("Sharpe (rf = 0)", f"{sharpe_ratio(portfolio):.2f}"),
            ("Máximo drawdown", f"{max_drawdown(portfolio):.2%}"),
            (
                "Backtest de VaR (Kupiec)",
                f"{bt.violations}/{bt.observations} violações · p-valor "
                f"{bt.kupiec_pvalue:.3f} · "
                f"{'APROVADO' if bt.approved else 'REPROVADO'}",
            ),
        ]
        y = 0.86
        fig.text(
            0.07,
            y + 0.025,
            "Métricas da carteira",
            fontsize=12,
            fontweight="bold",
            color=_INK,
        )
        for label, value in lines:
            y -= 0.026
            fig.text(0.09, y, label, fontsize=9.5, color=_MUTED)
            fig.text(0.55, y, value, fontsize=9.5, color=_INK, fontweight="bold")

        def _table(ax, df, title_text):
            ax.axis("off")
            ax.set_title(
                title_text,
                fontsize=11,
                fontweight="bold",
                color=_INK,
                loc="left",
                pad=14,
            )
            data = df.round(2).astype(str)
            # índice como primeira coluna, para nada vazar da página
            rows = zip(data.index, data.values.tolist(), strict=True)
            cell_text = [[str(i), *row] for i, row in rows]
            col_labels = [data.index.name or ""] + list(data.columns)
            table = ax.table(
                cellText=cell_text,
                colLabels=col_labels,
                loc="upper center",
                cellLoc="right",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.45)
            table.auto_set_column_width(list(range(len(col_labels))))
            for (_row, col), cell in table.get_celld().items():
                cell.set_edgecolor("#2A2F3E")
                cell.set_facecolor(_SURFACE)
                cell.set_text_props(color=_INK, ha="left" if col == 0 else "right")

        ax1 = fig.add_axes([0.07, 0.44, 0.86, 0.22])
        _table(ax1, report, f"VaR e CVaR pelos 3 métodos ({horizon_days} dias)")
        ax2 = fig.add_axes([0.07, 0.12, 0.86, 0.22])
        _table(ax2, stress, "Stress testing — cenários históricos (beta = 1)")
        fig.text(
            0.07,
            0.05,
            "Documento educacional gerado por código aberto — "
            "não é recomendação de investimento.",
            fontsize=8,
            color=_MUTED,
        )
        pdf.savefig(fig)
        plt.close(fig)

        # ---- Página 2: patrimônio e drawdown ----
        curve = equity_curve(portfolio, initial=100.0)
        dd = drawdown_series(portfolio) * 100
        fig, (ax_top, ax_bot) = plt.subplots(
            2,
            1,
            figsize=(8.27, 11.69),
            sharex=True,
            gridspec_kw={"height_ratios": [1.6, 1], "hspace": 0.1},
        )
        ax_top.plot(curve.index, curve, color=_LIME, linewidth=1.6)
        ax_top.set_title(
            "Curva de patrimônio (base 100)", loc="left", fontsize=12, fontweight="bold"
        )
        ax_bot.fill_between(dd.index, dd, 0, color=_RED, alpha=0.35, linewidth=0)
        ax_bot.plot(dd.index, dd, color=_RED, linewidth=1.0)
        ax_bot.set_title("Drawdown (%)", loc="left", fontsize=12, fontweight="bold")
        fig.subplots_adjust(left=0.09, right=0.96, top=0.94, bottom=0.05)
        pdf.savefig(fig)
        plt.close(fig)

        # ---- Página 3: Monte Carlo ----
        var_mc, cvar_mc = monte_carlo_var(portfolio, confidence, horizon_days)
        mu, sigma = portfolio.mean(), portfolio.std(ddof=1)
        rng = np.random.default_rng(42)
        terminal = rng.normal(mu, sigma, size=(10_000, horizon_days)).sum(axis=1) * 100
        fig, ax = plt.subplots(figsize=(8.27, 5.5))
        ax.hist(
            terminal,
            bins=60,
            color=_LIME,
            alpha=0.85,
            edgecolor=_SURFACE,
            linewidth=0.5,
        )
        ax.axvline(-var_mc * 100, color=_AMBER, linewidth=1.6)
        ax.axvline(-cvar_mc * 100, color=_RED, linewidth=1.6)
        ax.text(
            -var_mc * 100,
            ax.get_ylim()[1] * 0.95,
            f" VaR {confidence:.0%} = {var_mc:.1%}",
            color=_AMBER,
            fontsize=9,
        )
        ax.text(
            -cvar_mc * 100,
            ax.get_ylim()[1] * 0.87,
            f" CVaR = {cvar_mc:.1%}",
            color=_RED,
            fontsize=9,
            ha="right",
        )
        ax.set_title(
            f"Monte Carlo — P&L simulado em {horizon_days} dias " "(10.000 cenários)",
            loc="left",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("P&L acumulado (%)")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    return output
