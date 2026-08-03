"""
Rule Engine Test Suite

Purpose:
    Validate all RuleEngine logic using mocked ML predictions.

Run:
    python tests/test_rule_engine.py
"""

from services.rule_engine import rule_engine


class MockObject:
    def __init__(self, label):
        self.label = label


class MockVisionAnalysis:
    def __init__(self):
        self.face_matched = True
        self.detected_objects = []
        self.face_count = 1
        self.gaze_direction = "CENTER"


class MockAudioAnalysis:
    def __init__(self):
        self.is_human_speech = False


class MockPrediction:
    def __init__(self):
        self.vision_analysis = MockVisionAnalysis()
        self.audio_analysis = MockAudioAnalysis()
        self.system_status = "SUCCESS"
        self.continuous_violations = []
        self.snapshot_violations = []


class RuleEngineTester:

    def __init__(self):
        self.total = 0
        self.passed = 0

    def run(self, name, prediction, expected_violations):
        self.total += 1

        result = rule_engine.evaluate(prediction)

        actual = set(result["violations"])
        expected = set(expected_violations)

        if actual == expected:
            self.passed += 1
            print(f"[PASS] {name}")
        else:
            print(f"[FAIL] {name}")
            print(f"Expected : {expected}")
            print(f"Actual   : {actual}")
            print()

    def summary(self):
        print("\n" + "=" * 60)
        print("Rule Engine Test Summary")
        print("=" * 60)
        print(f"Tests Passed : {self.passed}/{self.total}")

        if self.passed == self.total:
            print("Overall      : PASS ✅")
        else:
            print("Overall      : FAIL ❌")

        print("=" * 60)


tester = RuleEngineTester()

# ==========================================================
# Test 1 : Clean Candidate
# ==========================================================

prediction = MockPrediction()

tester.run(
    "Clean Candidate",
    prediction,
    [],
)

# ==========================================================
# Test 2 : Face Authentication Failure
# ==========================================================

prediction = MockPrediction()
prediction.vision_analysis.face_matched = False

tester.run(
    "Face Authentication",
    prediction,
    [
        "Face authentication failed",
    ],
)

# ==========================================================
# Test 3 : Mobile Phone Detection
# ==========================================================

prediction = MockPrediction()
prediction.vision_analysis.detected_objects.append(
    MockObject("phone")
)

tester.run(
    "Mobile Phone Detection",
    prediction,
    [
        "Mobile phone detected",
    ],
)

# ==========================================================
# Test 4 : Multiple Persons
# ==========================================================

prediction = MockPrediction()
prediction.vision_analysis.face_count = 2

tester.run(
    "Multiple Persons",
    prediction,
    [
        "Multiple persons detected",
    ],
)

# ==========================================================
# Test 5 : Looking Left
# ==========================================================

prediction = MockPrediction()
prediction.vision_analysis.gaze_direction = "LEFT"

tester.run(
    "Looking Left",
    prediction,
    [
        "Candidate looking away",
    ],
)

# ==========================================================
# Test 6 : Looking Right
# ==========================================================

prediction = MockPrediction()
prediction.vision_analysis.gaze_direction = "RIGHT"

tester.run(
    "Looking Right",
    prediction,
    [
        "Candidate looking away",
    ],
)

# ==========================================================
# Test 7 : Human Speech
# ==========================================================

prediction = MockPrediction()
prediction.audio_analysis.is_human_speech = True

tester.run(
    "Speech Detection",
    prediction,
    [
        "Speech detected",
    ],
)

# ==========================================================
# Test 8 : Multiple Violations
# ==========================================================

prediction = MockPrediction()

prediction.vision_analysis.face_matched = False
prediction.vision_analysis.face_count = 2
prediction.vision_analysis.gaze_direction = "LEFT"
prediction.audio_analysis.is_human_speech = True
prediction.vision_analysis.detected_objects.append(
    MockObject("phone")
)

tester.run(
    "Multiple Violations",
    prediction,
    [
        "Face authentication failed",
        "Mobile phone detected",
        "Multiple persons detected",
        "Candidate looking away",
        "Speech detected",
    ],
)

tester.summary()
