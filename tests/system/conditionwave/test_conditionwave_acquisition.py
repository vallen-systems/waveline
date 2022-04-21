import asyncio
from time import perf_counter

import pytest

# all test coroutines will be treated as marked
pytestmark = pytest.mark.asyncio

@pytest.mark.repeat(10)
@pytest.mark.parametrize("channel", (1, 2))
async def test_acq_stream_pause(cw, channel):
    stream = cw.stream(channel, blocksize=1000)

    async def consume_n_blocks(n: int):
        block_count = 0
        async for _ in stream:
            block_count += 1
            if block_count >= n:
                break

    await cw.start_acquisition()

    await consume_n_blocks(10)

    # stop acq should not close stream port
    await cw.stop_acquisition()
    await cw.start_acquisition()

    await consume_n_blocks(10)

    await cw.stop_acquisition()
    await stream.aclose()


@pytest.mark.parametrize("decimation", (1, 2, 5, 10, 50, 100))
@pytest.mark.parametrize("channel", (1, 2))
async def test_acq_stream_decimation(cw, channel, decimation, duration_acq):
    samplerate = cw.MAX_SAMPLERATE / decimation
    block_size = 10_000
    block_duration = block_size / samplerate
    block_count_total = int(duration_acq / block_duration)
    stream = cw.stream(channel, block_size)

    await cw.set_tr_decimation(channel, decimation)
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
    await stream.aclose()

    assert time_elapsed == pytest.approx(duration_acq, rel=0.05)

