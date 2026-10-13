"""Live state for the v2 panel: training processes, GPUs, the experiment queue. Process / GPU introspection is
the legacy backend's (app/backend/main.py), reused as is; this module only shapes it for the v2 pages."""
from __future__ import annotations

import time

from .data import LOGS


def _legacy():
    from .. import main as legacy
    return legacy


_live_cache: dict = {"t": 0.0, "ids": set()}


def live_exp_ids() -> set[str]:
    """exp_ids of running training processes (cached 3 s: one page load asks for many batches)."""
    if time.time() - _live_cache["t"] > 3:
        lg = _legacy()
        _live_cache.update(t=time.time(), ids=lg._live_exp_ids(lg._ps_snapshot()))
    return _live_cache["ids"]


def driver_alive(man: dict | None) -> bool:
    return bool(man) and _legacy()._driver_alive(man)


def train_pct(batch: str, exp_id: str) -> float | None:
    return _legacy()._log_train_pct(LOGS / batch, exp_id)


def report_stage(batch: str) -> dict | None:
    return _legacy()._report_progress(batch)


def eval_state(batch: str, man: dict) -> dict:
    return _legacy()._eval_batch_state(batch, man)


def birth_time(paths) -> float | None:
    return _legacy()._birth_time(paths)


def system() -> dict:
    return _legacy().get_system()


def queue() -> dict:
    return _legacy().get_queue()


def gpu_history(since: float) -> dict:
    return _legacy().get_gpu_history(since)
