import asyncio
from time import perf_counter

import pytest


@pytest.mark.repeat(10)
@pytest.mark.parametrize("channel", [1, 2])
async def test_acq_stream_pause(lw, channel):
    await lw.set_tr_decimation(0, 1)
    stream = lw.stream(channel, blocksize=1000)

    async def consume_n_blocks(n: int):
        block_count = 0
        try:
            async for _ in stream:
                block_count += 1
                if block_count >= n:
                    break
        except ConnectionAbortedError:
            ...

    await lw.start_acquisition()

    await consume_n_blocks(100)

    # stop acq should not close stream port
    await lw.stop_acquisition()
    await lw.start_acquisition()

    await consume_n_blocks(100)

    await lw.stop_acquisition()
    await stream.aclose()


@pytest.mark.parametrize("decimation", [1, 2, 5, 10, 50, 100])
@pytest.mark.parametrize("channel", [1, 2])
async def test_acq_stream_decimation(lw, channel, decimation, duration_acq):
    samplerate = lw.MAX_SAMPLERATE / decimation
    block_size = 10_000
    block_duration = block_size / samplerate
    block_count_total = int(duration_acq / block_duration)
    stream = lw.stream(channel, block_size)

    await lw.set_tr_decimation(channel, decimation)
    await lw.start_acquisition()

    block_count = 0
    time_start = perf_counter()
    async for _ in stream:
        block_count += 1
        if block_count >= block_count_total:
            break
    time_stop = perf_counter()
    time_elapsed = time_stop - time_start

    await lw.stop_acquisition()
    await stream.aclose()

    assert time_elapsed == pytest.approx(duration_acq, rel=0.05)


@pytest.mark.parametrize("status_interval_seconds", [0, 0.1, 0.2])
@pytest.mark.parametrize("channel", [1, 2])
async def test_acq_only_status(lw, channel, status_interval_seconds):
    await lw.set_channel(0, False)  # disable all channels
    await lw.set_channel(channel, True)  # enable selected channel
    await lw.set_threshold(0, 10_000_000)  # above range
    await lw.set_continuous_mode(0, False)
    await lw.set_status_interval(0, status_interval_seconds)
    await lw.start_acquisition()
    await asyncio.sleep(1)
    await lw.stop_acquisition()

    ae_data = await lw.get_ae_data()
    if status_interval_seconds == 0:
        assert len(ae_data) == 0
    else:
        assert len(ae_data) == pytest.approx(1 / status_interval_seconds, abs=1)

    for i, record in enumerate(ae_data, start=1):
        assert record.channel == channel
        assert record.time == pytest.approx(i * status_interval_seconds, rel=0.05)
        assert record.type_ == "S"
        assert record.duration == pytest.approx(status_interval_seconds, rel=0.05)
        assert record.trai == 0


@pytest.mark.parametrize("channel", [1, 2])
async def test_acq_continuous_mode(lw, channel):
    ddt = 10_000  # 10 ms
    decimation = 1000  # prevent buffer overflows
    acq_duration = 1.0
    expected_hit_count = 100
    expected_samples = (ddt / 1e6) * (10e6 / decimation)

    await lw.set_channel(0, False)  # disable all channels
    await lw.set_channel(channel, True)  # enable selected channel
    await lw.set_status_interval(0, 1000)  # disable status data
    await lw.set_continuous_mode(0, True)
    await lw.set_ddt(0, ddt)
    await lw.set_tr_enabled(0, True)
    await lw.set_tr_decimation(0, decimation)

    await lw.start_acquisition()
    await asyncio.sleep(acq_duration)
    await lw.stop_acquisition()
    await asyncio.sleep(0.1)

    ae_data = await lw.get_ae_data()
    tr_data = await lw.get_tr_data()

    assert len(ae_data) == len(tr_data)
    assert len(ae_data) == pytest.approx(expected_hit_count, abs=2)
    for record in ae_data:
        assert record.trai != 0
    for record in tr_data:
        assert record.trai != 0
        assert record.samples == expected_samples


@pytest.mark.xfail(reason="only available since firmware version 2.13")
@pytest.mark.parametrize("channel", [0, 1, 2])
@pytest.mark.parametrize("samples", [0, 1_000, 1_000_000])
@pytest.mark.parametrize("pretrigger_samples", [0, 1_000])
async def test_acq_tr_snapshot(lw, channel, samples, pretrigger_samples):
    await lw.set_tr_decimation(0, 1)
    records = await lw.get_tr_snapshot(channel, samples, pretrigger_samples)

    if channel == 0:
        assert len(records) == 2
        assert records[0].channel == 1
        assert records[1].channel == 2
    else:
        assert len(records) == 1
        assert records[0].channel == channel

    for record in records:
        assert record.trai == 0
        assert record.time == 0
        assert record.samples == samples + pretrigger_samples


@pytest.mark.parametrize("count", [2, 4])
@pytest.mark.parametrize("interval", [0.1, 0.5])
@pytest.mark.parametrize("channel", [1, 2])
async def test_pulsing(lw, channel, interval, count):
    await lw.set_channel(0, False)  # disable all channels
    await lw.set_channel(channel, True)  # enable selected channel
    await lw.set_status_interval(0, 1000)  # disable status data
    await lw.set_continuous_mode(0, False)
    await lw.set_ddt(0, 1000)
    await lw.set_threshold(0, 10_000)
    await lw.set_tr_enabled(0, True)
    await lw.set_tr_decimation(0, 100)

    await lw.start_acquisition()
    await lw.start_pulsing(channel, interval, count)
    await asyncio.sleep(count * interval + 0.1)
    await lw.stop_acquisition()

    ae_data = await lw.get_ae_data()
    tr_data = await lw.get_tr_data()

    assert len(ae_data) == count
    assert len(tr_data) == count


@pytest.mark.parametrize("interval", [0.1, 0.5])
@pytest.mark.parametrize("channel", [1, 2])
async def test_stop_infinite_pulsing(lw, channel, interval):
    acq_time = 1.0
    expected_pulse_count = (acq_time - 2 * 0.02) / interval

    await lw.set_channel(0, False)  # disable all channels
    await lw.set_channel(channel, True)  # enable selected channel
    await lw.set_status_interval(0, 1000)  # disable status data
    await lw.set_continuous_mode(0, False)
    await lw.set_ddt(0, 1000)
    await lw.set_threshold(0, 10_000)
    await lw.set_tr_enabled(0, False)

    await lw.start_acquisition()
    await lw.start_pulsing(channel, interval, 0)
    await asyncio.sleep(acq_time)
    await lw.stop_pulsing()
    await asyncio.sleep(acq_time)  # now no new hits should be generated
    await lw.stop_acquisition()

    ae_data = await lw.get_ae_data()
    tr_data = await lw.get_tr_data()

    assert len(ae_data) == pytest.approx(expected_pulse_count, abs=1)
    assert len(tr_data) == 0
