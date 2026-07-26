from __future__ import annotations
import threading
from collections import deque
from dataclasses import dataclass


@dataclass
class StageConfig:
    threshold: float
    persistence: int = 2          # KEEP = 2 (your best setting)
    severity_mult: float = 1.0    # KEEP = 1.0 (your best setting)
    recheck_interval: int | None = None
    recheck_on_unknown: int = 8
    recheck_on_fp: int = 13
    recheck_on_tp: int = 6


class EscalationGate:
    def __init__(self, stage_configs: dict[str, StageConfig]):
        self.configs = stage_configs
        self._lock = threading.Lock()
        self._history = {s: deque(maxlen=cfg.persistence) for s, cfg in stage_configs.items()}
        self._episode_active = {s: False for s in stage_configs}
        self._windows_since_check = {s: 0 for s in stage_configs}
        self._last_verdict = {s: "Unknown" for s in stage_configs}
        self._grace_count = {s: 0 for s in stage_configs}

    def set_last_verdict(self, stage: str, verdict: str) -> None:
        with self._lock:
            self._last_verdict[stage] = verdict

    def update(self, stage: str, score: float) -> bool:
        with self._lock:
            cfg = self.configs[stage]
            hist = self._history[stage]

            if score <= cfg.threshold:
                if self._episode_active[stage] and self._grace_count[stage] < 1:
                    self._grace_count[stage] += 1
                    return False

                hist.clear()
                self._episode_active[stage] = False
                self._windows_since_check[stage] = 0
                self._last_verdict[stage] = "Unknown"
                self._grace_count[stage] = 0
                return False

            self._grace_count[stage] = 0
            hist.append(score)

            if self._episode_active[stage]:
                # If a hard recheck interval is provided, use it.
                # Otherwise, use adaptive recheck intervals based on the last verdict.
                effective_recheck = cfg.recheck_interval
                if effective_recheck is None:
                    last_verdict = self._last_verdict.get(stage, "Unknown")
                    if last_verdict == "Unknown":
                        effective_recheck = cfg.recheck_on_unknown
                    elif last_verdict == "False Positive":
                        effective_recheck = cfg.recheck_on_fp
                    elif last_verdict == "True Positive":
                        effective_recheck = cfg.recheck_on_tp

                if effective_recheck is None:
                    return False

                self._windows_since_check[stage] += 1
                if self._windows_since_check[stage] >= effective_recheck:
                    self._windows_since_check[stage] = 0
                    return True
                return False

            # Optional severity filter: only gate the initial trigger.
            # At 1.0 (default) this is a no-op. Set severity_mult > 1.0 to require
            # stronger anomalies before opening a new episode.
            if cfg.severity_mult > 1.0 and score <= cfg.threshold * cfg.severity_mult:
                return False

            if cfg.persistence < 1:
                return False

            persistent = len(hist) == cfg.persistence and all(s > cfg.threshold for s in hist)
            if persistent:
                self._episode_active[stage] = True
                self._windows_since_check[stage] = 0
                return True

            return False
        
    def reset(self, stage: str | None = None) -> None:
        with self._lock:
            for s in ([stage] if stage else self.configs.keys()):
                self._history[s].clear()
                self._episode_active[s] = False
                self._windows_since_check[s] = 0
                self._last_verdict[s] = "Unknown"
                self._grace_count[s] = 0