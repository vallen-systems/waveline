import asyncio

import pytest
from waveline import LinWave


@pytest.fixture()
async def lw(linwave_ip):
    def get_ip():
        if linwave_ip:
            return linwave_ip

        devices = LinWave.discover()
        if not devices:
            raise RuntimeError(
                "No linWave devices found. Please connect a device to run the system tests"
            )
        return devices[0]

    async with LinWave(get_ip()) as lw_:
        yield lw_

    await asyncio.sleep(0.05)  # wait for connection to be closed
