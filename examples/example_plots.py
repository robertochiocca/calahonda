"""Gera os gráficos do README a partir do próprio módulo.

Uso::

    pip install matplotlib
    python examples/example_plots.py

Salva três PNGs em ``docs/img/``: distribuição de retornos com VaR,
curva de patrimônio com drawdown, e simulação de Monte Carlo.
Funciona offline (dados sintéticos).
"""

import sys
from pathlib import Path

# Permite rodar o exemplo direto do clone, sem `pip install`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from calahonda_var import (
    drawdown_series,
    equity_curve,
    historical_var,
    load_returns,
    portfolio_returns,
)

OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "img"

# Paleta do site (design system da landing page)
SURFACE = "#0D0F14"
PANEL = "#141720"
INK = "#C8CDD8"
MUTED = "#8892A4"
GRID = "#2A2F3E"
LIME = "#C8F135"
AMBER = "#F5A623"
RED = "#F24B4B"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.grid": True,
    "axes.axisbelow": True,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
})


def _style(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_distribution(portfolio) -> None:
    """Histograma dos retornos diários com a normal ajustada e o VaR 95%."""
    var_1d, _ = historical_var(portfolio, confidence=0.95)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(
        portfolio * 100, bins=60, color=LIME, alpha=0.85,
        edgecolor=SURFACE, linewidth=0.6, density=True,
    )
    x = np.linspace(portfolio.min(), portfolio.max(), 300) * 100
    normal = stats.norm.pdf(x, portfolio.mean() * 100, portfolio.std(ddof=1) * 100)
    ax.plot(x, normal, color=MUTED, linewidth=1.4, linestyle="--")
    ax.text(0.985, 0.9, "– – normal ajustada", transform=ax.transAxes,
            color=MUTED, ha="right", fontsize=9)
    ax.axvline(-var_1d * 100, color=AMBER, linewidth=1.6)
    ax.text(-var_1d * 100 - 0.15, ax.get_ylim()[1] * 0.92,
            f"VaR 95% = {var_1d:.2%}", color=AMBER, ha="right", fontsize=9)
    ax.set_title("Distribuição dos retornos diários da carteira")
    ax.set_xlabel("Retorno diário (%)")
    ax.set_ylabel("Densidade")
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "distribuicao.png", dpi=150)
    plt.close(fig)


def plot_equity_drawdown(portfolio) -> None:
    """Curva de patrimônio (base 100) e drawdown, em dois painéis."""
    curve = equity_curve(portfolio, initial=100.0)
    dd = drawdown_series(portfolio) * 100
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 5), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.12},
    )
    ax1.plot(curve.index, curve, color=LIME, linewidth=1.6)
    ax1.set_title("Curva de patrimônio e drawdown (carteira 1/N)")
    ax1.set_ylabel("Patrimônio (base 100)")
    ax2.fill_between(dd.index, dd, 0, color=RED, alpha=0.35, linewidth=0)
    ax2.plot(dd.index, dd, color=RED, linewidth=1.0)
    ax2.text(0.02, 0.08, f"máx. drawdown = {abs(dd.min()):.1f}%",
             transform=ax2.transAxes, color=RED, fontsize=9,
             ha="left", va="bottom")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_ylim(dd.min() * 1.35, 2)
    for ax in (ax1, ax2):
        _style(ax)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.93, bottom=0.09)
    fig.savefig(OUT_DIR / "patrimonio_drawdown.png", dpi=150)
    plt.close(fig)


def plot_monte_carlo(portfolio, horizon_days: int = 21, n_paths: int = 120) -> None:
    """Trajetórias simuladas de Monte Carlo e a distribuição no horizonte."""
    mu = portfolio.mean()
    sigma = portfolio.std(ddof=1)
    rng = np.random.default_rng(42)
    steps = rng.normal(mu, sigma, size=(10_000, horizon_days))
    terminal = steps.sum(axis=1) * 100
    var_pct = -np.quantile(terminal, 0.05)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(8, 4.2), sharey=True,
        gridspec_kw={"width_ratios": [2, 1], "wspace": 0.06},
    )
    days = np.arange(horizon_days + 1)
    paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(steps[:n_paths], axis=1) * 100], axis=1
    )
    ax1.plot(days, paths.T, color=LIME, alpha=0.16, linewidth=0.8)
    ax1.plot(days, np.median(paths, axis=0), color=LIME, linewidth=1.8)
    ax1.axhline(-var_pct, color=AMBER, linewidth=1.4)
    ax1.text(0.4, -var_pct - 0.8, f"VaR 95% = {var_pct:.1f}%",
             color=AMBER, fontsize=9, va="top")
    ax1.set_title(f"Monte Carlo — {n_paths} de 10.000 trajetórias")
    ax1.set_xlabel("Dias úteis")
    ax1.set_ylabel("P&L acumulado (%)")
    ax1.set_xlim(0, horizon_days)

    ax2.hist(terminal, bins=60, orientation="horizontal", color=LIME,
             alpha=0.85, edgecolor=SURFACE, linewidth=0.5, density=True)
    ax2.axhline(-var_pct, color=AMBER, linewidth=1.4)
    ax2.set_title(f"Dia {horizon_days}")
    ax2.set_xlabel("Densidade")
    for ax in (ax1, ax2):
        _style(ax)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.9, bottom=0.13)
    fig.savefig(OUT_DIR / "monte_carlo.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    portfolio = portfolio_returns(load_returns())
    plot_distribution(portfolio)
    plot_equity_drawdown(portfolio)
    plot_monte_carlo(portfolio)
    print(f"3 gráficos salvos em {OUT_DIR}")


if __name__ == "__main__":
    main()
