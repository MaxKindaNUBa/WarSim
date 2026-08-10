"""Background-thread CSV writer so the training loop's CUDA work never blocks
on disk I/O.

The training loop just calls .log(row) — a non-blocking queue.put — and moves
straight on to the next env.step()/train_step(). A single background thread
pulls rows off that queue and appends them to disk on its own schedule. Since
file I/O releases the GIL, this genuinely overlaps disk writes with whatever
the main thread (and the GPU, which it drives) is doing next.
"""
import csv
import queue
import threading
from pathlib import Path

_SENTINEL = object()


class AsyncCSVLogger:
    def __init__(self, path, fieldnames):
        self.path = Path(path)
        self.fieldnames = fieldnames
        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()

        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def log(self, row):
        """Non-blocking: hand a row (dict matching fieldnames) to the writer thread."""
        self._queue.put(row)

    def _run(self):
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            while True:
                row = self._queue.get()
                if row is _SENTINEL:
                    break
                writer.writerow(row)
                f.flush()

    def close(self, timeout=5.0):
        """Block until every row queued so far has been written, then stop the thread."""
        self._queue.put(_SENTINEL)
        self._thread.join(timeout=timeout)
