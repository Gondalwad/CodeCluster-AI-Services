MOBILE_PHONE_WARNING = "Mobile phone detected"


class RuleEngine:
    """
    Converts ML violation codes into business-friendly violations.
    No ML decision logic should exist here.
    """

    VIOLATION_MAP = {
        # LOOKING_AWAY removed — gaze tracking too restrictive for real exam use
        "HEAD_TURNED": "Head turned",

        "FACE_MISSING": "Face not visible",
        "IDENTITY_MISMATCH": "Face authentication failed",
        "MULTIPLE_FACES": "Multiple persons detected",

        "SPEECH_DETECTED": "Speech detected",

        "SPECTACLES_DETECTED": "Spectacles detected",
    }

    OBJECT_MAP = {
        "PHONE": MOBILE_PHONE_WARNING,
        "CELL_PHONE": MOBILE_PHONE_WARNING,
        "MOBILE_PHONE": MOBILE_PHONE_WARNING,

        "BOOK": "Book detected",
        "LAPTOP": "Laptop detected",
        "TABLET": "Tablet detected",
        "EARPHONE": "Earphones detected",
        "HEADPHONE": "Headphones detected",
    }

    def evaluate(self, prediction):

        # ======================================================
        # Ignore invalid ML frames
        # ======================================================

        if prediction.system_status != "SUCCESS":

            return {
                "ignore_frame": True,

                "violations": [],
                "warning": False,
                "terminate": False,

                "system_status": prediction.system_status,

                "continuous_violations": [],
                "snapshot_violations": [],
                "ml_violations": [],
            }

        # ======================================================
        # Normal Processing
        # ======================================================

        violations = []

        # ------------------------------------------
        # ML Generated Violations
        # ------------------------------------------

        for violation in prediction.violations:

            # Normal ML violations
            if violation in self.VIOLATION_MAP:
                violations.append(
                    self.VIOLATION_MAP[violation]
                )
                continue

            # Object detection
            if violation.startswith("BANNED_OBJECT:"):

                obj = violation.split(":", 1)[1]

                violations.append(
                    self.OBJECT_MAP.get(
                        obj,
                        f"Banned object detected ({obj})"
                    )
                )

        # Remove duplicates while preserving order
        violations = list(dict.fromkeys(violations))

        return {
            "ignore_frame": False,

            "violations": violations,
            "warning": len(violations) > 0,
            "terminate": False,

            "system_status": prediction.system_status,

            "continuous_violations": list(
                prediction.continuous_violations
            ),

            "snapshot_violations": list(
                prediction.snapshot_violations
            ),

            # Raw ML violations (very useful for debugging)
            "ml_violations": list(
                prediction.violations
            ),
        }


rule_engine = RuleEngine()
