"""Wall-clock timing that is correct for asynchronous GPU execution.

CUDA/ROCm kernels are launched asynchronously, so time.perf_counter() alone can
stop before the GPU has finished. We synchronize the device immediately before
reading the clock at both the start and the end of a measured region.
"""
from __future__ import annotations

import time
from typing import Callable

import torch


def device_synchronizer(device: torch.device) -> Callable[[], None]:
    if device.type == "cuda":  # ROCm devices also report type "cuda"
        return lambda: torch.cuda.synchronize(device)
    return lambda: None


class SyncTimer:
    """Context manager: `with SyncTimer(sync) as t: ...; t.seconds`.

    `sync` and `clock` are injectable so the ordering can be unit-tested.
    """

    def __init__(self, sync: Callable[[], None], clock: Callable[[], float] = time.perf_counter):
        self._sync = sync
        self._clock = clock
        self._start: float | None = None
        self.seconds: float | None = None

    def __enter__(self) -> "SyncTimer":
        self._sync()                 # drain previously queued work so it is not counted
        self._start = self._clock()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._sync()                 # wait for work issued inside the block
        self.seconds = self._clock() - self._start
