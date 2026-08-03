"""
Warning Policy Configuration
"""

# Maximum warnings allowed before terminating exam
MAX_WARNINGS = 3

# Clean frames needed before a violation state resets (at 10 FPS, 5 = 0.5s)
MAX_MISSES = 5

# Cooldown (seconds) before the SAME violation can warn again
WARNING_COOLDOWN = 20

# Duration (seconds) the warning popup stays visible
WARNING_POPUP_DURATION = 2

# ==========================================================
# Violation Thresholds (consecutive frames before warning)
# ==========================================================

WARNING_THRESHOLDS = {

    # Head Pose — ~0.8s of sustained turn
    "Head turned": 8,

    # Identity
    "Face authentication failed": 15,

    # Face Visibility
    "Face not visible": 6,

    # Objects — 0.3s of visibility = warning
    "Mobile phone detected": 3,
    "Book detected": 3,
    "Laptop detected": 3,
    "Tablet detected": 3,
    "Earphones detected": 3,
    "Headphones detected": 3,

    # Multiple People
    "Multiple persons detected": 4,

    # Audio
    "Speech detected": 5,

    # Accessories
    "Spectacles detected": 4,

    # Fallback for any dynamic/unknown violation label
    "__default__": 4,
}
