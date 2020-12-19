import asyncio
from typing import NamedTuple
from unittest.mock import call, patch

from asynctest import CoroutineMock, Mock  # alternative to unittest.mock.AsyncMock in Python >=3.8
import pytest

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
        call(b"set_filter 0 0\n"),
    ]
    writer.write.assert_has_calls(expected_calls, any_order=True)


async def test_get_info(mock_objects):
    cw, reader, writer = mock_objects
    await cw.get_info()
    writer.write.assert_called_with(b"get_info\n")
    reader.read.assert_awaited()


@pytest.mark.parametrize(
    "value, command",
    (
        (0.05, b"set_adc_range 0 0\n"),
        (5, b"set_adc_range 0 1\n"),
    )
)
async def test_set_range(mock_objects, value, command):
    cw, _, writer = mock_objects
    await cw.set_range(value)
    writer.write.assert_called_with(command)
    assert cw.input_range == value


async def test_set_range_invalid(mock_objects):
    cw, *_ = mock_objects
    with pytest.raises(ValueError):
        await cw.set_range(0.1)


@pytest.mark.parametrize(
    "value, command",
    (
        (1, b"set_decimation 0 1\n"),
        (100, b"set_decimation 0 100\n"),
    )
)
async def test_set_decimation(mock_objects, value, command):
    cw, _, writer = mock_objects
    await cw.set_decimation(value)
    writer.write.assert_called_with(command)
    assert cw.decimation == value


@pytest.mark.parametrize("value", (-1, 0, 1000))
async def test_set_decimation_invalid(mock_objects, value):
    cw, *_ = mock_objects
    with pytest.raises(ValueError):
        await cw.set_decimation(value)


@pytest.mark.parametrize(
    "highpass, lowpass, order, command",
    (
        (None, None, 4, b"set_filter 0 0\n"),
        (0, 0, 0, b"set_filter 0 0.0 0.0 0\n"),
        (10e3, 350e3, 8, b"set_filter 0 10.0 350.0 8\n"),
        (10e3, None, 8, b"set_filter 0 10.0 5000.0 8\n"),
    )
)
async def test_set_filter(mock_objects, highpass, lowpass, order, command):
    cw, _, writer = mock_objects
    await cw.set_filter(highpass, lowpass, order)
    writer.write.assert_called_with(command)
    assert cw.filter_settings.highpass == highpass
    assert cw.filter_settings.lowpass == lowpass
    assert cw.filter_settings.order == order


async def test_start_stop_acquisition(mock_objects):
    cw, _, writer = mock_objects
    await cw.start_acquisition()
    writer.write.assert_called_with(b"start\n")
    await cw.stop_acquisition()
    writer.write.assert_called_with(b"stop\n")


async def test_acquisition_status(mock_objects):
    cw, reader, _ = mock_objects
    reader.readuntil.side_effect = [
        b"temp=20\n",
        b"buffer_size=1024\n",
        asyncio.CancelledError(),  # prevents raising StopIteration exception
    ]

    await cw.start_acquisition()
    await asyncio.sleep(0.1)  # wait until launched acquisition status task reads values
    assert cw.get_temperature() == 20
    assert cw.get_buffersize() == 1024
    await cw.stop_acquisition()
