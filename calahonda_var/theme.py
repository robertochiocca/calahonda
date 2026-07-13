"""Tema visual da Calahonda — paleta e estilo compartilhados.

Uma única fonte de verdade para as cores usadas na landing page, nos
gráficos matplotlib (README e PDF) e no dashboard Streamlit/Plotly.
"""

from __future__ import annotations

# Superfícies e texto
SURFACE = "#0D0F14"
PANEL = "#141720"
GRID = "#2A2F3E"
INK = "#C8CDD8"
MUTED = "#8892A4"

# Cores de dado
LIME = "#C8F135"  # série principal
BLUE = "#3B6EF8"  # série de comparação (benchmark)
AMBER = "#F5A623"  # limiar de atenção (VaR)
RED = "#F24B4B"  # limiar crítico (CVaR, drawdown)
GREEN = "#3ECF6E"  # status positivo

MATPLOTLIB_RC: dict = {
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
}

PLOTLY_LAYOUT: dict = {
    "paper_bgcolor": SURFACE,
    "plot_bgcolor": SURFACE,
    "font": {"color": INK, "size": 12},
    "colorway": [LIME, BLUE, AMBER, RED],
    "xaxis": {"gridcolor": GRID, "zerolinecolor": GRID},
    "yaxis": {"gridcolor": GRID, "zerolinecolor": GRID},
    "margin": {"l": 50, "r": 20, "t": 40, "b": 40},
    "hoverlabel": {"bgcolor": PANEL, "font": {"color": INK}},
}
