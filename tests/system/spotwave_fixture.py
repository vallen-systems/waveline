import pytest

from waveline import SpotWave


@pytest.fixture(scope="module")
def sw():
    devices = SpotWave.discover()
    if not devices:
        raise RuntimeError(
            "No SpotWave devices found. Please connect a device to run the system tests"
        )
    with SpotWave(devices[0]) as sw:
        yield sw
