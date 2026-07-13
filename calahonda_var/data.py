"""Carregamento de dados: B3 via yfinance, com fallback sintético offline."""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_TICKERS: tuple[str, ...] = ("PETR4.SA", "VALE3.SA", "ITUB4.SA")

# Parâmetros diários plausíveis para ações brasileiras (usados no fallback):
# volatilidade entre ~1.6% e ~2.2% ao dia e correlação positiva moderada.
_SYNTH_MU = np.array([-0.0004, -0.0002, 0.0002])
_SYNTH_VOL = np.array([0.022, 0.020, 0.016])
_SYNTH_CORR = 0.35


def synthetic_returns(
    n_days: int = 756,
    tickers: tuple[str, ...] = DEFAULT_TICKERS,
    seed: int = 7,
) -> pd.DataFrame:
    """Gera retornos diários sintéticos correlacionados.

    Permite rodar todo o projeto offline, sem depender de rede nem de
    ``yfinance``. Com a mesma ``seed`` o resultado é sempre idêntico.

    Parameters
    ----------
    n_days : int
        Número de dias úteis (756 ≈ 3 anos). Padrão 756.
    tickers : tuple[str, ...]
        Nomes das colunas do DataFrame resultante.
    seed : int
        Semente do gerador, para reprodutibilidade.
    """
    if n_days < 2:
        raise ValueError(f"`n_days` deve ser >= 2, recebeu {n_days}.")
    n_assets = len(tickers)
    if n_assets == 0:
        raise ValueError("`tickers` não pode ser vazio.")

    rng = np.random.default_rng(seed)

    # Recicla os parâmetros base caso haja mais (ou menos) de 3 ativos.
    mu = np.resize(_SYNTH_MU, n_assets)
    vol = np.resize(_SYNTH_VOL, n_assets)

    corr = np.full((n_assets, n_assets), _SYNTH_CORR)
    np.fill_diagonal(corr, 1.0)
    cov = corr * np.outer(vol, vol)

    data = rng.multivariate_normal(mean=mu, cov=cov, size=n_days)
    index = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    return pd.DataFrame(data, index=index, columns=list(tickers))


def load_returns(
    tickers: tuple[str, ...] = DEFAULT_TICKERS,
    period: str = "3y",
    fallback: bool = True,
) -> pd.DataFrame:
    """Baixa preços da B3 via yfinance e devolve retornos diários simples.

    Se ``yfinance`` não estiver instalado ou a rede falhar, cai no
    gerador sintético (a menos que ``fallback=False``).
    """
    try:
        import logging

        import yfinance as yf

        # Sem rede, o yfinance loga erros próprios antes de levantar a
        # exceção; silencia para o fallback ser limpo.
        logging.getLogger("yfinance").setLevel(logging.CRITICAL)
        prices = yf.download(
            list(tickers),
            period=period,
            progress=False,
            auto_adjust=True,
            timeout=10,
        )["Close"]
        returns = prices.pct_change().dropna(how="any")
        if returns.empty:
            raise ValueError(f"yfinance não retornou dados para {tickers}.")
        return returns
    except Exception:
        if not fallback:
            raise
        return synthetic_returns(tickers=tickers)


def _parse_portfolio_frame(df: pd.DataFrame) -> tuple[tuple[str, ...], np.ndarray]:
    """Valida e normaliza um DataFrame de carteira (colunas ticker/peso)."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if not {"ticker", "peso"}.issubset(df.columns):
        raise ValueError(
            f"CSV precisa das colunas 'ticker' e 'peso', recebeu {list(df.columns)}."
        )
    df = df.dropna(subset=["ticker", "peso"])
    if df.empty:
        raise ValueError("CSV não tem nenhuma linha válida de carteira.")

    tickers = tuple(
        t if "." in t else f"{t}.SA"
        for t in df["ticker"].astype(str).str.strip().str.upper()
    )
    weights = df["peso"].astype(float).to_numpy()
    if (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Pesos devem ser não-negativos e somar mais que zero.")
    return tickers, weights / weights.sum()


def load_portfolio_file(
    source, name: str | None = None
) -> tuple[tuple[str, ...], np.ndarray]:
    """Importa uma carteira de CSV, Excel (.xlsx) ou JSON.

    O arquivo precisa das colunas ``ticker`` e ``peso`` (JSON: lista de
    objetos com essas chaves). Pesos em fração (0.4) ou porcentagem (40)
    são normalizados para somar 1; tickers sem sufixo ganham ``.SA``.

    Parameters
    ----------
    source : str | Path | file-like
        Caminho ou buffer do arquivo.
    name : str | None
        Nome do arquivo, para detectar a extensão quando ``source`` é um
        buffer (ex.: upload do Streamlit). ``None`` usa ``source.name``.

    Returns
    -------
    (tickers, weights)
        Tickers prontos para o yfinance e pesos normalizados.
    """
    filename = (name or getattr(source, "name", str(source))).lower()
    extension = filename.rsplit(".", 1)[-1]
    if extension == "csv":
        frame = pd.read_csv(source)
    elif extension in ("xlsx", "xls"):
        frame = pd.read_excel(source)
    elif extension == "json":
        frame = pd.read_json(source)
    else:
        raise ValueError(
            f"Formato não suportado: .{extension} (aceitos: csv, xlsx, json)."
        )
    return _parse_portfolio_frame(frame)


def load_portfolio_csv(path_or_buffer) -> tuple[tuple[str, ...], np.ndarray]:
    """Importa uma carteira de um CSV (atalho para ``load_portfolio_file``)."""
    return _parse_portfolio_frame(pd.read_csv(path_or_buffer))


def load_benchmark(period: str = "3y", fallback: bool = True) -> pd.Series:
    """Retornos diários do Ibovespa (^BVSP) via yfinance.

    Sem rede (ou sem ``yfinance``), devolve um benchmark sintético: a
    carteira de pesos iguais dos ativos sintéticos padrão — coerente com
    o fallback de ``load_returns``.
    """
    try:
        import logging

        import yfinance as yf

        logging.getLogger("yfinance").setLevel(logging.CRITICAL)
        prices = yf.download(
            "^BVSP", period=period, progress=False, auto_adjust=True, timeout=10
        )["Close"]
        returns = prices.pct_change().dropna()
        if isinstance(returns, pd.DataFrame):
            returns = returns.iloc[:, 0]
        if returns.empty:
            raise ValueError("yfinance não retornou dados para ^BVSP.")
        returns.name = "IBOV"
        return returns
    except Exception:
        if not fallback:
            raise
        benchmark = portfolio_returns(synthetic_returns())
        benchmark.name = "IBOV (sintético)"
        return benchmark


def portfolio_returns(returns: pd.DataFrame, weights=None) -> pd.Series:
    """Retorno da carteira a partir dos retornos por ativo.

    Parameters
    ----------
    returns : pandas.DataFrame
        Retornos simples, um ativo por coluna.
    weights : array-like | None
        Pesos por ativo. ``None`` usa pesos iguais; pesos que não somam
        1 são normalizados.
    """
    n_assets = returns.shape[1]
    if weights is None:
        w = np.full(n_assets, 1.0 / n_assets)
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.size != n_assets:
            raise ValueError(f"`weights` tem {w.size} pesos para {n_assets} ativos.")
        if w.sum() <= 0:
            raise ValueError("A soma dos pesos deve ser positiva.")
        w = w / w.sum()
    return pd.Series(returns.to_numpy() @ w, index=returns.index, name="portfolio")
