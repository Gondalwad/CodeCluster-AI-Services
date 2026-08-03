import time

from services.warning_manager import WarningManager
from config.warning_policy import WARNING_COOLDOWN


def test_same_violation_does_not_double_warn_without_clear():
    manager = WarningManager()
    candidate_id = "cand-1"

    for _ in range(5):
        result = manager.update(candidate_id, ["Candidate looking away"])

    assert result["warning_count"] == 1
    assert result["warning"] is True

    state = manager.sessions[candidate_id].violations["Candidate looking away"]
    state.last_warning_time = time.time() - WARNING_COOLDOWN - 1
    state.frames = 5
    state.warned = True

    result = manager.update(candidate_id, ["Candidate looking away"])

    assert result["warning_count"] == 1
    assert result["warning"] is False


def test_warning_counts_for_new_violation_cycle_after_clear():
    manager = WarningManager()
    candidate_id = "cand-2"

    for _ in range(5):
        result = manager.update(candidate_id, ["Candidate looking away"])

    assert result["warning_count"] == 1

    result = manager.update(candidate_id, [])
    assert result["warning_count"] == 1

    for _ in range(5):
        result = manager.update(candidate_id, ["Candidate looking away"])

    assert result["warning_count"] == 2
    assert result["warning"] is True
