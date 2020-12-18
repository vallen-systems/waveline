import pytest

from waveline import SpotWave


def pytest_addoption(parser):
    parser.addoption(
        "--duration-acq",
        help="Duration of acquistion in seconds for TR loss test",
        type=float,
        default=1,
    )


@pytest.fixture
def duration_acq(pytestconfig):
    return pytestconfig.getoption("duration_acq")


@pytest.fixture(scope="module")
def sw():
    devices = SpotWave.discover()
    if not devices:
        raise RuntimeError(
            "No SpotWave devices found. Please connect a device to run the system tests"
        )
    with SpotWave(devices[0]) as sw:
        sw.set_datetime()  # set current date/time
        sw.clear_buffer()
        yield sw
