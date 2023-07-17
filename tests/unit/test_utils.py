import pytest
from waveline.utils import decibel_to_volts, volts_to_decibel


@pytest.mark.parametrize(
    ("input_", "output"),
    [
        (0, 1e-6),
        (100, 0.1),
        ([100, 120, 140], [0.1, 1, 10]),
    ],
)
def test_decibel_to_volts(input_, output):
    assert decibel_to_volts(input_) == pytest.approx(output)


@pytest.mark.parametrize(
    ("input_", "output"),
    [
        (1e-6, 0),
        (0.1, 100),
        ([0.1, 1, 10], [100, 120, 140]),
    ],
)
def test_volts_to_decibel(input_, output):
    assert volts_to_decibel(input_) == pytest.approx(output)
