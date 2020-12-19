import asyncio
from time import perf_counter

import pytest


# all test coroutines will be treated as marked
pytestmark = pytest.mark.asyncio


async def test_connected(cw):
    assert cw.connected


async def test_get_info(cw):
    info = await cw.get_info()
    lines = info.strip().split("\n")
    info_dict = dict([line.split("=", maxsplit=1) for line in lines])
    keys_expected = [
        "dev_id",
        "fpga_id",
        "channel_count",
        "range_count",
        "max_sample_rate",
        "sw_version",
        "fpga_version",
    ]
    assert set(info_dict.keys()) == set(keys_expected)


@pytest.mark.parametrize("decimation", (1, 2, 5, 10, 50, 100))
@pytest.mark.parametrize("channel", (1, 2))
async def test_acq_decimation(cw, channel, decimation, duration_acq):
    block_size = 10_000
    samplerate = cw.MAX_SAMPLERATE / decimation
    block_count_total = int(duration_acq * samplerate / block_size)

    await cw.set_decimation(decimation)
    await cw.start_acquisition()

    block_count = 0
    time_start = perf_counter()
    async for _ in cw.stream(channel, block_size):
        block_count += 1
        if block_count >= block_count_total:
            break
    time_stop = perf_counter()
    time_elapsed = time_stop - time_start

    await cw.stop_acquisition()

    assert time_elapsed == pytest.approx(duration_acq, rel=0.025)


async def test_acq_status(cw):
    assert cw.get_temperature() == None
    assert cw.get_buffersize() == 0

    await cw.set_decimation(10)  # prevent buffer overflow, we don't read the data
    await cw.start_acquisition()

    async for _ in cw.stream(1, 10_000):
        await asyncio.sleep(2.5)  #  wait for acq status, sent every 2 seconds
        assert cw.get_temperature() != 0
        assert cw.get_buffersize() > 0
        break

    await cw.stop_acquisition()
