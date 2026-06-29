from __future__ import annotations

import time
from contextlib import contextmanager
from collections.abc import Iterator


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    state = {"elapsed": 0.0}
    start = time.perf_counter()
    try:
        yield state
    finally:
        state["elapsed"] = time.perf_counter() - start

