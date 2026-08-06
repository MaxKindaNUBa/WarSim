"""Per-episode CSV writer — appends one row every training episode."""
import csv
from pathlib import Path

FIELDNAMES = ["episode", "total_reward", "length", "outcome", "hit_rate", "ammo_used"]


class EpisodeLogger:
    def __init__(self, run_dir):
        self.path = Path(run_dir) / "episode_log.csv"
        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    def log(self, episode, total_reward, length, outcome, hit_rate, ammo_used):
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writerow({
                "episode": episode,
                "total_reward": total_reward,
                "length": length,
                "outcome": outcome,
                "hit_rate": hit_rate,
                "ammo_used": ammo_used,
            })
