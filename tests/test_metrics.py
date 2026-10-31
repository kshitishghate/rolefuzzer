"""Tests for custom Inspect metrics."""


from inspect_ai.scorer import SampleScore, Score

from rolefuzzer.metrics import full_violation_rate, mean_ttfv, violation_rate
from rolefuzzer.spec import STATUS_SEVERITY


def _make_sample_score(value: str, ttfv: int | None = None) -> SampleScore:
    """Create a SampleScore with numeric value (matching scorer output) and TTFV metadata."""
    return SampleScore(
        score=Score(
            value=STATUS_SEVERITY.get(value, 0),
            answer=value,
            metadata={"ttfv": ttfv},
        ),
    )


def _make_invalid_sample_score() -> SampleScore:
    return SampleScore(
        score=Score(
            value=0,
            answer="invalid",
            metadata={"invalid_sample": True, "ttfv": None},
        ),
    )


class TestViolationRate:
    def test_no_deviations(self):
        metric = violation_rate()
        scores = [_make_sample_score("full_adherence"), _make_sample_score("full_adherence")]
        assert metric(scores) == 0.0

    def test_all_deviations(self):
        metric = violation_rate()
        scores = [_make_sample_score("clear_breach"), _make_sample_score("partial_deviation")]
        assert metric(scores) == 1.0

    def test_mixed(self):
        metric = violation_rate()
        scores = [
            _make_sample_score("full_adherence"),
            _make_sample_score("partial_deviation"),
            _make_sample_score("clear_breach"),
            _make_sample_score("full_adherence"),
        ]
        assert metric(scores) == 0.5

    def test_empty(self):
        metric = violation_rate()
        assert metric([]) == 0.0

    def test_invalid_samples_excluded_from_denominator(self):
        metric = violation_rate()
        scores = [
            _make_sample_score("partial_deviation"),
            _make_sample_score("full_adherence"),
            _make_invalid_sample_score(),
        ]
        assert metric(scores) == 0.5


class TestFullViolationRate:
    def test_no_breaches(self):
        metric = full_violation_rate()
        scores = [_make_sample_score("full_adherence"), _make_sample_score("partial_deviation")]
        assert metric(scores) == 0.0

    def test_all_breaches(self):
        metric = full_violation_rate()
        scores = [_make_sample_score("clear_breach"), _make_sample_score("clear_breach")]
        assert metric(scores) == 1.0

    def test_mixed(self):
        metric = full_violation_rate()
        scores = [
            _make_sample_score("full_adherence"),
            _make_sample_score("clear_breach"),
            _make_sample_score("partial_deviation"),
            _make_sample_score("clear_breach"),
        ]
        assert metric(scores) == 0.5

    def test_invalid_samples_excluded_from_denominator(self):
        metric = full_violation_rate()
        scores = [
            _make_sample_score("clear_breach"),
            _make_sample_score("full_adherence"),
            _make_invalid_sample_score(),
        ]
        assert metric(scores) == 0.5


class TestMeanTTFV:
    def test_no_deviations(self):
        metric = mean_ttfv()
        scores = [_make_sample_score("full_adherence", ttfv=None)]
        assert metric(scores) == 0.0

    def test_single_breach(self):
        metric = mean_ttfv()
        scores = [_make_sample_score("clear_breach", ttfv=3)]
        assert metric(scores) == 3.0

    def test_average(self):
        metric = mean_ttfv()
        scores = [
            _make_sample_score("clear_breach", ttfv=2),
            _make_sample_score("partial_deviation", ttfv=4),
            _make_sample_score("full_adherence", ttfv=None),  # Excluded
        ]
        assert metric(scores) == 3.0

    def test_empty(self):
        metric = mean_ttfv()
        assert metric([]) == 0.0

    def test_invalid_samples_excluded(self):
        metric = mean_ttfv()
        scores = [
            _make_sample_score("clear_breach", ttfv=2),
            _make_invalid_sample_score(),
        ]
        assert metric(scores) == 2.0
