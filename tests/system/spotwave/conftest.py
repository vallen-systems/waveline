import pytest
from waveline import SpotWave


@pytest.fixture(scope="module")
def sw():
    devices = SpotWave.discover()
    if not devices:
        raise RuntimeError(
            "No spotWave devices found. Please connect a device to run the system tests"
        )
    with SpotWave(devices[0]) as sw_:
        try:
            yield sw_
        finally:
            sw_.clear_buffer()
            sw_.stop_acquisition()
            sw_.clear_data_log()
            # set reasonable defaults
            sw_.set_continuous_mode(False)
            sw_.set_logging_mode(False)
            sw_.set_tr_decimation(1)
