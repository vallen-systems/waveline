import asyncio
from typing import NamedTuple
from unittest.mock import call, patch

import pytest
from asynctest import CoroutineMock, Mock  # alternative to unittest.mock.AsyncMock in Python >=3.8

from waveline import ConditionWave

# all test coroutines will be treated as marked
pytestmark = pytest.mark.asyncio

wait_forever = lambda: asyncio.sleep(3600)


class MockedObjects(NamedTuple):
    cw: ConditionWave
    reader: CoroutineMock
    writer: CoroutineMock


@pytest.fixture(autouse=True, name="mock_objects")
async def mock_asyncio_connection():
    with patch("asyncio.open_connection", new=CoroutineMock()) as mock_create_connection:
        mock_reader = Mock(spec=asyncio.StreamReader)
        mock_writer = Mock(spec=asyncio.StreamWriter)
        mock_create_connection.return_value = (mock_reader, mock_writer)

        async with ConditionWave("192.168.0.100") as cw:
            yield MockedObjects(
                cw=cw,
                reader=mock_reader,
                writer=mock_writer,
            )


async def test_reconnect(mock_objects):
    cw, *_ = mock_objects
    assert cw.connected
    await cw.close()
    assert not cw.connected
    await cw.connect()
    assert cw.connected


async def test_default_settings(mock_objects):
    _, _, writer = mock_objects

    expected_calls = [
        call(b"set_adc_range 0 @0\n"),
        call(b"set_acq tr_decimation 1 @0\n"),
    ]
    writer.write.assert_has_calls(expected_calls, any_order=True)


async def test_get_info(mock_objects):
    cw, reader, writer = mock_objects
    reader.readline.side_effect = [
        b"fw_version=2.1\n",
        b"fpga_version=3.1\n",
        b"channel_count=2\n",
        b"range_count=2\n",
        b"max_sample_rate=10000000\n",
        b"adc2uv=1.5625 156.25\n",
        wait_forever(),
    ]
    info = await cw.get_info()
    writer.write.assert_called_with(b"get_info\n")
    reader.readline.assert_awaited()

    assert info.firmware_version == "2.1"
    assert info.fpga_version == "3.1"
    assert info.channel_count == 2
    assert info.range_count == 2
    assert info.max_sample_rate == 10_000_000
    assert info.adc_to_volts == [1.5625e-6, 156.25e-6]


async def test_get_status(mock_objects):
    cw, reader, writer = mock_objects
    reader.readline.side_effect = [
        b"temp=55\n",
        b"buffer_size=112014\n",
        wait_forever(),
    ]
    status = await cw.get_status()
    writer.write.assert_called_with(b"get_status\n")
    reader.readline.assert_awaited()

    assert status.temperature == 55
    assert status.buffer_size == 112014


async def test_get_setup(mock_objects):
    cw, reader, writer = mock_objects
    reader.readline.side_effect = [
        b"channel=1\n",
        b"dsp\n",
        b"adc_range=0\n",
        b"adc2uv=1.5625\n",
        b"filter=none - 300 kHz, order 8\n",
        b"ae\n",
        b"enabled=1\n",
        b"cont=0\n",
        b"thr=100.0 uV\n",
        b"ddt=500.0 us\n",
        b"status_interval=100 ms\n",
        b"tr\n",
        b"tr_enabled=1\n",
        b"tr_decimation=2\n",
        b"tr_pre_trig=100\n",
        b"tr_post_dur=50\n",
        b"\n",
        wait_forever(),
    ]
    setup = await cw.get_setup(1)
    writer.write.assert_called_with(b"get_setup @1\n")
    reader.readline.assert_awaited()

    assert setup.adc_range_volts == 0.05
    assert setup.adc_to_volts == 1.5625e-6
    assert setup.filter_highpass_hz == None
    assert setup.filter_lowpass_hz == 300e3
    assert setup.filter_order == 8
    assert setup.enabled == True
    assert setup.continuous_mode == False
    assert setup.threshold_volts == 100e-6
    assert setup.ddt_seconds == 500e-6
    assert setup.status_interval_seconds == 0.1
    assert setup.tr_enabled == True
    assert setup.tr_decimation == 2
    assert setup.tr_pretrigger_samples == 100
    assert setup.tr_postduration_samples == 50


@pytest.mark.parametrize(
    "channel, value, command",
    (
        (0, 0.05, b"set_adc_range 0 @0\n"),
        (1, 5, b"set_adc_range 1 @1\n"),
        (2, 5, b"set_adc_range 1 @2\n"),
    ),
)
async def test_set_range(mock_objects, channel, value, command):
    cw, _, writer = mock_objects
    await cw.set_range(channel, value)
    writer.write.assert_called_with(command)


@pytest.mark.parametrize("channel", (-1, 3))
async def test_set_range_invalid_channel(mock_objects, channel):
    with pytest.raises(ValueError):
        await mock_objects.cw.set_range(channel, 0.05)


@pytest.mark.parametrize("value", (-1, 0, 11))
async def test_set_range_invalid_value(mock_objects, value):
    with pytest.raises(ValueError):
        await mock_objects.cw.set_range(0, value)


@pytest.mark.parametrize(
    "channel, value, command",
    (
        (0, 1, b"set_acq tr_decimation 1 @0\n"),
        (1, 4, b"set_acq tr_decimation 4 @1\n"),
        (2, 8, b"set_acq tr_decimation 8 @2\n"),
        (0, 1000, b"set_acq tr_decimation 1000 @0\n"),
    ),
)
async def test_set_tr_decimation(mock_objects, channel, value, command):
    cw, _, writer = mock_objects
    await cw.set_tr_decimation(channel, value)
    writer.write.assert_called_with(command)


@pytest.mark.parametrize("channel", (-1, 3))
async def test_set_tr_decimation_invalid_channel(mock_objects, channel):
    with pytest.raises(ValueError):
        await mock_objects.cw.set_tr_decimation(channel, 1)


@pytest.mark.parametrize("value", (-1, 0, 1001))
async def test_set_tr_decimation_invalid_value(mock_objects, value):
    with pytest.raises(ValueError):
        await mock_objects.cw.set_tr_decimation(0, value)


@pytest.mark.parametrize(
    "channel, highpass, lowpass, order, command",
    (
        (0, None, None, 4, b"set_filter none none 4 @0\n"),
        (1, 0, 0, 0, b"set_filter 0.0 0.0 0 @1\n"),
        (1, 10e3, 350e3, 8, b"set_filter 10.0 350.0 8 @1\n"),
        (2, 10e3, None, 8, b"set_filter 10.0 none 8 @2\n"),
    ),
)
async def test_set_filter(mock_objects, channel, highpass, lowpass, order, command):
    cw, _, writer = mock_objects
    await cw.set_filter(channel, highpass, lowpass, order)
    writer.write.assert_called_with(command)


@pytest.mark.parametrize("channel", (-1, 3))
async def test_set_filter_invalid_channel(mock_objects, channel):
    with pytest.raises(ValueError):
        await mock_objects.cw.set_filter(channel, 50e3, 300e3, 4)


async def test_start_stop_acquisition(mock_objects):
    cw, _, writer = mock_objects
    await cw.start_acquisition()
    writer.write.assert_called_with(b"start_acq\n")
    await cw.stop_acquisition()
    writer.write.assert_called_with(b"stop_acq\n")
