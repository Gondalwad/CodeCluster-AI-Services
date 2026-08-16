import time
import logging
from dataclasses import dataclass, field

from config.warning_policy import (
    WARNING_THRESHOLDS,
    MAX_WARNINGS,
    MAX_MISSES,
    WARNING_COOLDOWN,
)

logger = logging.getLogger(__name__)


@dataclass
class ViolationState:
    frames: int = 0
    misses: int = 0
    warned: bool = False
    last_warning_time: float = 0.0


@dataclass
class CandidateState:
    warning_count: int = 0
    violations: dict = field(default_factory=dict)
    last_any_warning_time: float = 0.0
    cooldown_reset_done: bool = True


class WarningManager:
    def __init__(self):
        self.thresholds = dict(WARNING_THRESHOLDS)
        self.max_warnings = MAX_WARNINGS
        self.max_misses = MAX_MISSES
        self.sessions = {}

    def _get_candidate(self, candidate_id):
        if candidate_id not in self.sessions:
            self.sessions[candidate_id] = CandidateState(
                warning_count=0,
                violations={
                    name: ViolationState()
                    for name in self.thresholds
                    if name != "__default__"
                },
            )
        return self.sessions[candidate_id]

    def _reset_all_violations(self, candidate):
        for state in candidate.violations.values():
            state.frames = 0
            state.misses = 0
            state.warned = False
        logger.info("[COOLDOWN_RESET] All violation counters reset to 0")

    def update(self, candidate_id, current_violations):
        candidate = self._get_candidate(candidate_id)
        current = set(current_violations)

        default_threshold = self.thresholds.get("__default__", 4)
        for v in current:
            if v not in candidate.violations:
                candidate.violations[v] = ViolationState()
                self.thresholds.setdefault(v, default_threshold)

        now = time.time()

        global_cooldown_remaining = max(
            0.0,
            round(WARNING_COOLDOWN - (now - candidate.last_any_warning_time), 1)
        )
        in_global_cooldown = global_cooldown_remaining > 0

        if in_global_cooldown:
            candidate.cooldown_reset_done = False
            logger.debug("[WM COOLDOWN] candidate=%s | %.1fs remaining", candidate_id, global_cooldown_remaining)

            debug = {}
            for violation_name, state in candidate.violations.items():
                threshold = self.thresholds[violation_name]
                debug[violation_name] = {
                    "frames": state.frames,
                    "misses": state.misses,
                    "threshold": threshold,
                    "warned": state.warned,
                    "cooldown_remaining": global_cooldown_remaining,
                }

            terminate = (
                candidate.warning_count >= self.max_warnings
                and not False
            )

            return {
                "warning": False,
                "violations": [],
                "warning_count": candidate.warning_count,
                "terminate": terminate,
                "new_warnings": [],
                "warning_label": None,
                "warning_message": None,
                "global_cooldown_remaining": global_cooldown_remaining,
                "debug": debug,
            }

        if not candidate.cooldown_reset_done:
            self._reset_all_violations(candidate)
            candidate.cooldown_reset_done = True

        new_warnings = []
        debug = {}

        for violation_name, state in candidate.violations.items():
            threshold = self.thresholds[violation_name]
            detected = violation_name in current

            if detected:
                state.misses = 0
                state.frames += 1

                if state.frames >= threshold:
                    candidate.warning_count += 1
                    state.warned = True
                    state.last_warning_time = now
                    candidate.last_any_warning_time = now
                    state.frames = 0
                    new_warnings.append(violation_name)

            else:
                state.misses += 1
                if state.misses >= self.max_misses:
                    state.frames = 0
                    state.misses = 0

            debug[violation_name] = {
                "frames": state.frames,
                "misses": state.misses,
                "threshold": threshold,
                "warned": state.warned,
                "cooldown_remaining": 0.0,
            }

        warning_triggered = bool(new_warnings)
        warning_label = new_warnings[0] if new_warnings else None
        warning_message = (
            f"Warning {candidate.warning_count}: {warning_label}"
            if warning_label
            else None
        )

        terminate = (
            candidate.warning_count >= self.max_warnings
            and not warning_triggered
        )

        return {
            "warning": warning_triggered,
            "violations": new_warnings if warning_triggered else [],
            "warning_count": candidate.warning_count,
            "terminate": terminate,
            "new_warnings": new_warnings,
            "warning_label": warning_label,
            "warning_message": warning_message,
            "global_cooldown_remaining": global_cooldown_remaining,
            "debug": debug,
        }

    def reset(self, candidate_id):
        self.sessions.pop(candidate_id, None)


warning_manager = WarningManager()
