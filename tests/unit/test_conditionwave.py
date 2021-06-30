import asyncio
from typing import NamedTuple
from unittest.mock import call, patch

import pytest
from asynctest import CoroutineMock, Mock  # alternative to unittest.mock.AsyncMock in Python >=3.8

from waveline import ConditionWave

# all test coroutines will be treated as marked
pytestmark = pytest.mark.asyncio


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
        call(b"set_adc_range 0 0\n"),
        call(b"set_decimation 0 1\n"),
    ]
    writer.write.assert_has_calls(expected_calls, any_order=True)


async def test_get_info(mock_objects):
    cw, reader, writer = mock_objects
    await cw.get_info()
    writer.write.assert_called_with(b"get_info\n")
    reader.read.assert_awaited()


@pytest.mark.parametrize(
    "channel, value, command",
    (
        (0, 0.05, b"set_adc_range 0 0\n"),
        (1, 5, b"set_adc_range 1 1\n"),
        (2, 5, b"set_adc_range 2 1\n"),
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
        (0, 1, b"set_decimation 0 1\n"),
        (1, 4, b"set_decimation 1 4\n"),
        (2, 8, b"set_decimation 2 8\n"),
        (0, 100, b"set_decimation 0 100\n"),
    ),
)
async def test_set_decimation(mock_objects, channel, value, command):
    cw, _, writer = mock_objects
    await cw.set_decimation(channel, value)
    writer.write.assert_called_with(command)


@pytest.mark.parametrize("channel", (-1, 3))
async def test_set_decimation_invalid_channel(mock_objects, channel):
    with pytest.raises(ValueError):
        await mock_objects.cw.set_decimation(channel, 1)


@pytest.mark.parametrize("value", (-1, 0, 1000))
async def test_set_decimation_invalid_value(mock_objects, value):
    with pytest.raises(ValueError):
        await mock_objects.cw.set_decimation(0, value)


@pytest.mark.parametrize(
    "channel, highpass, lowpass, order, command",
    (
        (1, None, None, 4, b"set_filter 1 0\n"),
        (1, 0, 0, 0, b"set_filter 1 0.0 0.0 0\n"),
        (1, 10e3, 350e3, 8, b"set_filter 1 10.0 350.0 8\n"),
        (2, 10e3, None, 8, b"set_filter 2 10.0 5000.0 8\n"),
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
    writer.write.assert_called_with(b"start\n")
    await cw.stop_acquisition()
    writer.write.assert_called_with(b"stop\n")
