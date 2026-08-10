"""Sets up per-run logging (stdlib logging + tensorboard) in one call at run
start. Every other module just calls logging.getLogger(__name__) — nothing
else configures logging.
"""
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml
from torch.utils.tensorboard import SummaryWriter

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


def start_run(run_name, config, run_dir=None):
    """Create runs/<run_name>_<timestamp>/ (or the given run_dir verbatim — e.g.
    training/train_multi_seed.py passes runs/<experiment>_<timestamp>/<seed>/ so every
    seed's logs land in its own folder under one shared experiment folder), wire up
    run.log + console logging and a tensorboard SummaryWriter, and snapshot the
    fully-resolved config.

    Returns (run_dir: Path, writer: SummaryWriter).
    """
    if run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = RUNS_DIR / f"{run_name}_{timestamp}"
    else:
        run_dir = Path(run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "tensorboard").mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    file_handler = logging.FileHandler(run_dir / "run.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    with open(run_dir / "config_snapshot.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    return run_dir, writer
