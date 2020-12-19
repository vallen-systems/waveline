import pytest

from waveline import ConditionWave


@pytest.fixture
async def cw(cwave_ip):
    def get_ip():
        if cwave_ip:
            return cwave_ip

        devices = ConditionWave.discover()
        if not devices:
            raise RuntimeError(
                "No conditionWave devices found. Please connect a device to run the system tests"
            )
        return devices[0]

    cw = ConditionWave(get_ip())
    await cw.connect()
    yield cw
    await cw.close()
