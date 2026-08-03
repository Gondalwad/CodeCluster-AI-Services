import time
from dataclasses import dataclass, field

from config.warning_policy import (
    WARNING_THRESHOLDS,
    MAX_WARNINGS,
    MAX_MISSES,
    WARNING_COOLDOWN,
)


@dataclass
class ViolationState:
    """
    Stores the state of one violation.
    """

    # Consecutive violation frames
    frames: int = 0

    # Consecutive clean frames
    misses: int = 0

    # Warning already issued for current violation
    warned: bool = False

    # Timestamp of last warning
    last_warning_time: float = 0.0


@dataclass
class CandidateState:
    warning_count: int = 0
    violations: dict = field(default_factory=dict)


class WarningManager:

    def __init__(self):

        self.thresholds = WARNING_THRESHOLDS
        self.max_warnings = MAX_WARNINGS
        self.max_misses = MAX_MISSES

        self.sessions = {}

    # =======================================================
    # Candidate
    # =======================================================

    def _get_candidate(self, candidate_id):

        if candidate_id not in self.sessions:

            self.sessions[candidate_id] = CandidateState(
                warning_count=0,
                violations={
                    name: ViolationState()
                    for name in self.thresholds
                },
            )

        return self.sessions[candidate_id]

    # =======================================================
    # Update
    # =======================================================

    def update(self, candidate_id, current_violations):

        candidate = self._get_candidate(candidate_id)

        current = set(current_violations)

        # Ensure any new dynamic violation (e.g. "Banned object detected (X)")
        # gets a state entry on the fly using the __default__ threshold
        default_threshold = self.thresholds.get("__default__", 4)
        for v in current:
            if v not in candidate.violations:
                candidate.violations[v] = ViolationState()
                self.thresholds.setdefault(v, default_threshold)

        now = time.time()

        new_warnings = []

        debug = {}

        for violation_name, state in candidate.violations.items():

            threshold = self.thresholds[violation_name]
            detected = violation_name in current

            if detected:
                # A new active violation cycle should start only after the
                # candidate was clean again. This prevents one long streak from
                # being treated as multiple human-facing warnings.
                state.misses = 0

                if not state.warned:
                    state.frames += 1

                cooldown_over = (
                    now - state.last_warning_time
                ) >= WARNING_COOLDOWN

                if (
                    not state.warned
                    and state.frames >= threshold
                    and cooldown_over
                ):
                    candidate.warning_count += 1
                    state.warned = True
                    state.last_warning_time = now
                    new_warnings.append(violation_name)
                    state.frames = 0

            else:
                state.misses += 1

                if state.misses >= self.max_misses:
                    state.frames = 0
                    state.misses = 0
                    state.warned = False

            debug[violation_name] = {
                "frames": state.frames,
                "misses": state.misses,
                "threshold": threshold,
                "warned": state.warned,
                "cooldown_remaining": max(
                    0,
                    round(
                        WARNING_COOLDOWN
                        - (now - state.last_warning_time),
                        2,
                    ),
                ),
            }

        warning_triggered = bool(new_warnings)

        warning_label = new_warnings[0] if new_warnings else None
        warning_message = (
            f"Warning {candidate.warning_count}: {warning_label}"
            if warning_label
            else None
        )

        return {
            "warning": warning_triggered,
            "violations": new_warnings if warning_triggered else [],
            "warning_count": candidate.warning_count,
            "terminate": candidate.warning_count >= self.max_warnings,
            "new_warnings": new_warnings,
            "warning_label": warning_label,
            "warning_message": warning_message,
            "debug": debug,
        }

    # =======================================================
    # Reset
    # =======================================================

    def reset(self, candidate_id):

        self.sessions.pop(candidate_id, None)


warning_manager = WarningManager()
