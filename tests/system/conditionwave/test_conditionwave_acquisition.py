import asyncio
from time import perf_counter

import pytest

# all test coroutines will be treated as marked
pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("decimation", (1, 2, 5, 10, 50, 100))
@pytest.mark.parametrize("channel", (1,))
async def test_acq_decimation(cw, channel, decimation, duration_acq):
    samplerate = cw.MAX_SAMPLERATE / decimation
    block_size = 10_000
    block_duration = block_size / samplerate
    block_count_total = int(duration_acq / block_duration)
    stream = cw.stream(channel, block_size)

    await cw._send_command("set_acq enabled 1")
    await cw._send_command("set_acq tr_enabled 1")
    await cw.set_tr_decimation(channel, decimation)
    await cw.stop_acquisition()
    await cw.start_acquisition()

    block_count = 0
    time_start = perf_counter()
    async for _ in stream:
        block_count += 1
        if block_count >= block_count_total:
            break
    time_stop = perf_counter()
    time_elapsed = time_stop - time_start

    await cw.stop_acquisition()

    assert time_elapsed == pytest.approx(duration_acq, rel=0.025)
