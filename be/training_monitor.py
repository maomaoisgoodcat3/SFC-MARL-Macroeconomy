import time
import logging
from collections import deque
from typing import Optional, Deque
import numpy as np

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    import GPUtil
    _HAS_GPUTIL = True
except ImportError:
    _HAS_GPUTIL = False

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

logger = logging.getLogger("InstitutionalEconomist.TrainingMonitor")


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class TrainingMonitor:
    """
    Theo doi HIEU SUAT VAN HANH cua qua trinh train (khong dung de danh gia
    chat luong kinh te cua policy -- viec do da co custom_metrics/Parquet
    rieng): tieu thu phan cung (CPU/RAM/GPU neu co), toc do (giay/iteration),
    ETA toi iteration muc tieu, va MOT TIN HIEU THAM KHAO THO ve xu huong
    reward gan day (doc hoi quy tuyen tinh don gian).

    QUAN TRONG ve gioi han: "xu huong reward" o day KHONG PHAI mot bao dam
    hoi tu hinh thuc. Deep MARL noi chung khong co bao dam hoi tu toan cuc
    ngoai tabular setting (Rawat, P. (2026), "A Survey of Reinforcement
    Learning for Economics", arXiv:2603.08956, phan Limitations: "the
    absence of global convergence guarantees outside of tabular settings").
    Day chi la mot chi bao nhanh cho nguoi van hanh theo doi trong luc train,
    khong phai mot claim khoa hoc de trich dan trong ket qua.
    """
    TREND_WINDOW = 10

    def __init__(self, total_target_iters: int, console: Optional[Console] = None):
        self.total_target_iters = total_target_iters
        self.console = console or Console()
        self.start_time = time.time()
        self._iter_durations: Deque[float] = deque(maxlen=self.TREND_WINDOW)
        self._reward_history: Deque[float] = deque(maxlen=self.TREND_WINDOW)
        self._last_tick = self.start_time

        if _HAS_PSUTIL:
            psutil.cpu_percent(interval=None)  # goi "mo" de lay baseline, gia tri dau bo qua

    def tick(self, mean_return: float) -> None:
        """Goi 1 lan/iteration, NGAY SAU khi co train_results tu algo.train()."""
        now = time.time()
        self._iter_durations.append(now - self._last_tick)
        self._last_tick = now
        if np.isfinite(mean_return):
            self._reward_history.append(float(mean_return))

    def _convergence_trend(self) -> str:
        if len(self._reward_history) < 4:
            return "chua du du lieu"
        y = np.array(self._reward_history, dtype=np.float64)
        x = np.arange(len(y), dtype=np.float64)
        slope = float(np.polyfit(x, y, 1)[0])
        std = float(np.std(y)) if len(y) > 1 else 0.0
        if std < 1e-6 or abs(slope * len(y)) < 0.5 * std:
            return "[yellow]~ Ổn định / Plateau[/yellow]"
        elif slope > 0:
            return f"[green]↑ Đang cải thiện (dốc +{slope:.1f}/iter)[/green]"
        else:
            return f"[red]↓ Đang giảm (dốc {slope:.1f}/iter)[/red]"

    def _hardware_snapshot(self) -> str:
        parts = []
        if _HAS_PSUTIL:
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
            parts.append(f"CPU {cpu:4.1f}%")
            parts.append(f"RAM {ram:4.1f}%")
        else:
            parts.append("CPU/RAM: psutil chưa cài")

        if _HAS_GPUTIL:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    g = gpus[0]
                    parts.append(f"GPU {g.load * 100:4.1f}% (VRAM {g.memoryUtil * 100:4.1f}%)")
                else:
                    parts.append("GPU: không phát hiện (đang chạy CPU-only, num_gpus=0)")
            except Exception:
                parts.append("GPU: không đọc được")
        return "  |  ".join(parts)

    def render_summary(self, current_iter: int) -> None:
        """In 1 panel tom tat -- goi DINH KY (vd. moi 5 iter), khong goi moi
        iteration de tranh loang man hinh voi dong [TRAIN] plain-text da co
        (dong do van can giu de grep/redirect ra file log)."""
        avg_iter_sec = float(np.mean(self._iter_durations)) if self._iter_durations else 0.0
        remaining_iters = max(0, self.total_target_iters - current_iter)
        eta_seconds = avg_iter_sec * remaining_iters
        elapsed_seconds = time.time() - self.start_time
        progress_pct = (current_iter / max(1, self.total_target_iters)) * 100.0

        table = Table.grid(padding=(0, 2))
        table.add_column(justify="left", style="bold cyan", no_wrap=True)
        table.add_column(justify="left")
        table.add_row("Tiến độ", f"{current_iter}/{self.total_target_iters} iter ({progress_pct:.1f}%)")
        table.add_row("Tốc độ", f"{avg_iter_sec:.1f}s/iter (TB {len(self._iter_durations)} iter gần nhất)")
        table.add_row("Đã chạy", _format_duration(elapsed_seconds))
        table.add_row("ETA tới target", _format_duration(eta_seconds) if remaining_iters > 0 else "Đã xong")
        table.add_row("Phần cứng", self._hardware_snapshot())
        table.add_row("Xu hướng reward", self._convergence_trend())

        try:
            self.console.print(Panel(table, title=f"[bold]Training Monitor — Iter {current_iter}[/bold]", border_style="blue"))
        except Exception as exc:
            logger.warning(f"[WARNING] Could not render rich summary panel: {str(exc)}")
