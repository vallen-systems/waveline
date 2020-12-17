import pytest
from pytest import mark
from spotwave_fixture import sw


@mark.parametrize("samples", (
    10,
    1024,
    65536,
    200_000,
))
def test_get_data(sw, samples):
    data = sw.get_data(samples)
    assert len(data) == samples
