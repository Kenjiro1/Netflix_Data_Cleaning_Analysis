"""
utils.py — Resource Monitoring for AutoML vs Manual Pipeline Experiment
========================================================================
Autor: Experimento Científico AutoML vs Manual
Descrição: Módulo robusto para monitoramento de memória RSS e CPU
           usando psutil com amostragem em thread dedicada.

Uso reprodutível: todos os métodos são thread-safe e determinísticos.
"""

import time
import threading
import warnings
from typing import Optional, Callable, Any, Tuple

import matplotlib
import numpy as np
import pandas as pd
import psutil
from scipy import integrate


class ResourceMonitor:
    """
    Monitor de recursos do sistema baseado em thread dedicada.

    Captura o Resident Set Size (RSS) do processo atual em intervalos
    regulares, armazenando timestamps e valores de memória para análise
    posterior. Fornece métodos para calcular métricas de custo agregado.

    Parameters
    ----------
    interval : float
        Intervalo de amostragem em segundos (padrão: 0.1 = 100ms).
    process : psutil.Process, optional
        Processo a monitorar. Se None, usa o processo atual.

    Examples
    --------
    >>> monitor = ResourceMonitor(interval=0.1)
    >>> with monitor:
    ...     train_model(X, y)
    >>> df = monitor.to_dataframe()
    >>> peak = monitor.peak_memory_mb()
    >>> auc = monitor.memory_time_auc()
    """

    def __init__(
        self,
        interval: float = 0.1,
        process: Optional[psutil.Process] = None,
    ):
        self.interval = interval
        self._process = process or psutil.Process()

        # Armazenamento de amostras (acesso thread-safe via lock)
        self._lock = threading.Lock()
        self._timestamps: list[float] = []
        self._rss_bytes: list[int] = []

        # Controle da thread de monitoramento
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_wall: Optional[float] = None
        self._end_wall: Optional[float] = None

    # ------------------------------------------------------------------
    # Context manager interface
    # ------------------------------------------------------------------

    def __enter__(self) -> "ResourceMonitor":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Controle de ciclo de vida
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Inicia a thread de monitoramento e registra t=0."""
        if self._thread is not None and self._thread.is_alive():
            warnings.warn("Monitor já está em execução. Ignorando start().")
            return

        # Reset completo
        with self._lock:
            self._timestamps.clear()
            self._rss_bytes.clear()
        self._stop_event.clear()
        self._start_wall = time.perf_counter()

        # Captura uma amostra imediata (t=0) antes de lançar a thread
        self._sample()

        self._thread = threading.Thread(
            target=self._monitoring_loop,
            name="ResourceMonitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Para a thread de monitoramento e captura amostra final."""
        if self._thread is None:
            return

        self._stop_event.set()
        self._thread.join(timeout=self.interval * 10)
        self._end_wall = time.perf_counter()

        # Amostra final garantida
        self._sample()

    def _monitoring_loop(self) -> None:
        """Loop principal da thread de monitoramento."""
        while not self._stop_event.is_set():
            self._sample()
            self._stop_event.wait(timeout=self.interval)

    def _sample(self) -> None:
        """Captura uma amostra de RSS e armazena com timestamp relativo."""
        try:
            rss = self._process.memory_info().rss
            ts = time.perf_counter() - (self._start_wall or time.perf_counter())
            with self._lock:
                self._timestamps.append(ts)
                self._rss_bytes.append(rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass  # Processo encerrou durante monitoramento

    # ------------------------------------------------------------------
    # Propriedades de conveniência
    # ------------------------------------------------------------------

    @property
    def elapsed_seconds(self) -> float:
        """Tempo total decorrido entre start() e stop() em segundos."""
        if self._start_wall is None:
            return 0.0
        end = self._end_wall or time.perf_counter()
        return end - self._start_wall

    # ------------------------------------------------------------------
    # Exportação de dados
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """
        Retorna as amostras como um DataFrame pandas.

        Returns
        -------
        pd.DataFrame
            Colunas: ['timestamp_s', 'rss_bytes', 'rss_mb']
            Ordenado por timestamp_s crescente.
        """
        with self._lock:
            ts = list(self._timestamps)
            rss = list(self._rss_bytes)

        df = pd.DataFrame({
            "timestamp_s": ts,
            "rss_bytes": rss,
        })
        df["rss_mb"] = df["rss_bytes"] / (1024 ** 2)
        df.sort_values("timestamp_s", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def to_csv(self, path: str) -> None:
        """Salva as amostras em CSV."""
        self.to_dataframe().to_csv(path, index=False)

    # ------------------------------------------------------------------
    # Métricas derivadas
    # ------------------------------------------------------------------

    def peak_memory_mb(self) -> float:
        """Pico de memória RSS em MiB durante o monitoramento."""
        with self._lock:
            if not self._rss_bytes:
                return 0.0
            return max(self._rss_bytes) / (1024 ** 2)

    def mean_memory_mb(self) -> float:
        """Média de memória RSS em MiB durante o monitoramento."""
        df = self.to_dataframe()
        return float(df["rss_mb"].mean()) if not df.empty else 0.0

    def memory_time_auc(self, method: str = "trapezoid") -> float:
        """
        Área sob a curva de memória ao longo do tempo (MiB·s).

        Representa o 'custo total de memória no tempo': quanto mais alta
        a curva e mais longa a execução, maior o valor.

        Parameters
        ----------
        method : str
            Método de integração numérica: 'trapezoid' (padrão) ou 'simpson'.

        Returns
        -------
        float
            Integral em MiB·s (megabytes × segundos).
        """
        df = self.to_dataframe()
        if len(df) < 2:
            return 0.0

        x = df["timestamp_s"].values
        y = df["rss_mb"].values

        if method == "trapezoid":
            return float(integrate.trapezoid(y, x))
        elif method == "simpson":
            # Simpson exige n ímpar; usa trapezoid como fallback se necessário
            try:
                return float(integrate.simpson(y, x=x))
            except ValueError:
                return float(integrate.trapezoid(y, x))
        else:
            raise ValueError(f"Método desconhecido: {method!r}. Use 'trapezoid' ou 'simpson'.")

    def summary(self) -> dict:
        """
        Retorna um dicionário com as métricas principais do monitoramento.

        Returns
        -------
        dict
            Chaves: elapsed_s, peak_memory_mb, mean_memory_mb,
                    memory_time_auc_mib_s, n_samples
        """
        df = self.to_dataframe()
        return {
            "elapsed_s": round(self.elapsed_seconds, 4),
            "peak_memory_mb": round(self.peak_memory_mb(), 2),
            "mean_memory_mb": round(self.mean_memory_mb(), 2),
            "memory_time_auc_mib_s": round(self.memory_time_auc(), 2),
            "n_samples": len(df),
        }


# ---------------------------------------------------------------------------
# Decorator de conveniência
# ---------------------------------------------------------------------------

def monitor_resources(
    interval: float = 0.1,
) -> Callable:
    """
    Decorator que envolve uma função com ResourceMonitor.

    Retorna uma tupla (result, monitor) onde `monitor` contém todos os dados
    coletados.

    Parameters
    ----------
    interval : float
        Intervalo de amostragem em segundos.

    Examples
    --------
    >>> @monitor_resources(interval=0.1)
    ... def train():
    ...     model.fit(X_train, y_train)
    >>> result, mon = train()
    >>> print(mon.summary())
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Tuple[Any, ResourceMonitor]:
            monitor = ResourceMonitor(interval=interval)
            with monitor:
                result = func(*args, **kwargs)
            return result, monitor
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Função utilitária: gráfico comparativo (estilo IEEE/ACM)
# ---------------------------------------------------------------------------

def plot_memory_comparison(
    monitors: dict[str, ResourceMonitor],
    title: str = "Memory Consumption Over Time",
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (7.16, 4.5),  # largura de coluna dupla IEEE
) -> "matplotlib.figure.Figure":
    """
    Gera gráfico 'Memory vs. Time' com estilo de publicação IEEE/ACM.

    Parameters
    ----------
    monitors : dict[str, ResourceMonitor]
        Mapeamento {label: monitor} para cada pipeline.
    title : str
        Título do gráfico.
    output_path : str, optional
        Caminho para salvar a figura (PDF/PNG/SVG). Se None, apenas retorna.
    figsize : tuple
        Tamanho da figura em polegadas (largura, altura).

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    # --- Estilo IEEE/ACM ---------------------------------------------------
    IEEE_STYLE = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.7",
        "lines.linewidth": 1.5,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }

    # Paleta colorblind-safe (distinguível em P&B também)
    PALETTE = [
        "#E64B35",  # vermelho
        "#4DBBD5",  # azul claro
        "#00A087",  # verde-teal
        "#3C5488",  # azul escuro
        "#F39B7F",  # salmão
        "#8491B4",  # lilás
    ]
    LINESTYLES = ["-", "--", "-.", ":"]

    with matplotlib.rc_context(IEEE_STYLE):
        fig, ax = plt.subplots(figsize=figsize)

        for idx, (label, monitor) in enumerate(monitors.items()):
            df = monitor.to_dataframe()
            color = PALETTE[idx % len(PALETTE)]
            ls = LINESTYLES[idx % len(LINESTYLES)]

            ax.plot(
                df["timestamp_s"],
                df["rss_mb"],
                label=label,
                color=color,
                linestyle=ls,
                linewidth=1.8,
                zorder=3,
            )

            # Marca o pico
            peak_idx = df["rss_mb"].idxmax()
            ax.scatter(
                df.loc[peak_idx, "timestamp_s"],
                df.loc[peak_idx, "rss_mb"],
                color=color,
                s=40,
                zorder=5,
                marker="^",
            )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Resident Set Size (MiB)")
        ax.set_title(title, pad=8)

        # Tick minor
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
        ax.tick_params(which="minor", length=2, color="0.5")

        # Legenda fora do plot (estilo IEEE)
        ax.legend(
            loc="upper left",
            frameon=True,
            ncol=1,
        )

        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

        fig.tight_layout()

        if output_path:
            fig.savefig(output_path)
            print(f"[plot] Figura salva em: {output_path}")

    return fig


def build_comparison_dataframe(
    results: list[dict],
) -> pd.DataFrame:
    """
    Constrói o DataFrame de comparação final do experimento.

    Parameters
    ----------
    results : list of dict
        Cada dict deve conter:
          - 'Pipeline': nome do pipeline
          - 'monitor': instância ResourceMonitor já finalizada
          - 'accuracy': acurácia (ou outra métrica) do modelo

    Returns
    -------
    pd.DataFrame
        Colunas: [Pipeline, Tempo_Total_s, Pico_Memoria_MB,
                  Media_Memoria_MB, Area_Sob_Curva_MiBs, Acuracia]
    """
    rows = []
    for entry in results:
        mon: ResourceMonitor = entry["monitor"]
        s = mon.summary()
        rows.append({
            "Pipeline": entry["Pipeline"],
            "Tempo_Total_s": s["elapsed_s"],
            "Pico_Memoria_MB": s["peak_memory_mb"],
            "Media_Memoria_MB": s["mean_memory_mb"],
            "Area_Sob_Curva_MiBs": s["memory_time_auc_mib_s"],
            "Acuracia": round(entry.get("accuracy", float("nan")), 4),
            "N_Amostras_Monitor": s["n_samples"],
        })
    df = pd.DataFrame(rows)
    return df