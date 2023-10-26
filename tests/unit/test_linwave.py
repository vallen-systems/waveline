import asyncio
from typing import NamedTuple
from unittest.mock import call, patch

try:
    from unittest.mock import AsyncMock, Mock
except ImportError:
    from asyncmock import AsyncMock, Mock  # for Python < 3.8

import numpy as np
import pytest
from numpy.testing import assert_allclose
from waveline.linwave import Info, LinWave

CLOCK = 10e6
ADC_TO_VOLTS = [1.5625e-06, 0.00015625]
ADC_TO_EU = [factor**2 * 1e14 / CLOCK for factor in ADC_TO_VOLTS]


class MockedObjects(NamedTuple):
    lw: LinWave
    reader: AsyncMock
    writer: AsyncMock


@pytest.fixture(autouse=True, name="mock_objects")
async def mock_linwave_with_asyncio_connection():
    with patch("asyncio.open_connection", new=AsyncMock()) as mock_create_connection:
        mock_reader = Mock(spec=asyncio.StreamReader)
        mock_writer = Mock(spec=asyncio.StreamWriter)
        mock_create_connection.return_value = (mock_reader, mock_writer)

        lw = LinWave("192.168.0.100")

        try:
            with patch.object(LinWave, "get_info", new=AsyncMock()) as mock_get_info:
                mock_get_info.return_value = Info(
                    hardware_id="E8EB1B3D9E76",
                    firmware_version="2.13",
                    channel_count=2,
                    input_range=["50 mV", "5 V"],
                    adc_to_volts=[1.5625e-6, 156.25e-6],
                    extra={},
                )
                await lw.connect()  # get_info called during connect to get adc2uv, firmware, ...

            yield MockedObjects(lw, mock_reader, mock_writer)
        finally:
            await lw.close()


async def test_default_settings(mock_objects):
    _, _, writer = mock_objects

    expected_calls = [
        call(b"set_adc_range 0 @0\n"),
        call(b"set_acq tr_decimation 1 @0\n"),
    ]
    writer.write.assert_has_calls(expected_calls, any_order=True)


async def test_identify(mock_objects):
    lw, _, writer = mock_objects
    await lw.identify(channel=2)
    writer.write.assert_called_with(b"identify @2\n")


async def test_get_info(mock_objects):
    lw, reader, writer = mock_objects
    reader.readline.side_effect = [
        b"fw_version=2.1\n",
        b"fpga_version=3.1\n",
        b"channel_count=2\n",
        b"range_count=2\n",
        b"max_sample_rate=10000000\n",
        b"adc2uv=1.5625 156.25\n",
        b"\n",
        TimeoutError,
    ]
    info = await lw.get_info()
    writer.write.assert_called_with(b"get_info\n")
    reader.readline.assert_awaited()

    assert info.hardware_id is None
    assert info.firmware_version == "2.1"
    assert info.channel_count == 2
    assert info.input_range == ["50 mV", "5 V"]
    assert info.adc_to_volts == [1.5625e-6, 156.25e-6]
    assert {"fpga_version", "range_count", "max_sample_rate"} == info.extra.keys()


async def test_get_info_since_v2_13(mock_objects):
    lw, reader, writer = mock_objects
    reader.readline.side_effect = [
        b"hw_id=E8EB1B3D9E76",  # new
        b"fw_version=2.13\n",
        b"fpga_version=3.3\n",
        b"channel_count=2\n",
        b"input_range=50 mV, 5 V\n",  # range_count -> input ranges as strings
        b"max_samplerate=10000000\n",  # max_sample_rate -> max_samplerate
        b"adc2uv=1.5625, 156.25\n",  # comma as delimiter
        b"\n",
        TimeoutError,
    ]
    info = await lw.get_info()
    writer.write.assert_called_with(b"get_info\n")
    reader.readline.assert_awaited()

    assert info.hardware_id == "E8EB1B3D9E76"
    assert info.firmware_version == "2.13"
    assert info.channel_count == 2
    assert info.input_range == ["50 mV", "5 V"]
    assert info.adc_to_volts == [1.5625e-6, 156.25e-6]
    assert {"fpga_version", "max_samplerate"} == info.extra.keys()


async def test_get_status(mock_objects):
    lw, reader, writer = mock_objects
    reader.readline.side_effect = [
        b"temp=42\n",
        b"recording=1\n",
        b"pulsing=0\n",
        b"buffer_size=0\n",
        b"buffer_capacity=1024\n",
        b"\n",
        TimeoutError,
    ]
    status = await lw.get_status()
    writer.write.assert_called_with(b"get_status\n")
    reader.readline.assert_awaited()

    assert status.temperature == 42
    assert status.recording is True
    assert status.pulsing is False
    assert {"buffer_size", "buffer_capacity"} == status.extra.keys()


async def test_get_setup(mock_objects):
    lw, reader, writer = mock_objects
    reader.readline.side_effect = [
        b"channel=1\n",
        b"dsp\n",
        b"adc_range=1\n",
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
        TimeoutError,
    ]
    setup = await lw.get_setup(1)
    writer.write.assert_called_with(b"get_setup @1\n")
    reader.readline.assert_awaited()

    assert setup.enabled is True
    assert setup.input_range == 1
    assert setup.adc_to_volts == 1.5625e-6
    assert setup.filter_highpass_hz is None
    assert setup.filter_lowpass_hz == 300e3
    assert setup.filter_order == 8
    assert setup.continuous_mode is False
    assert setup.threshold_volts == 100e-6
    assert setup.ddt_seconds == 500e-6
    assert setup.status_interval_seconds == 0.1
    assert setup.tr_enabled is True
    assert setup.tr_decimation == 2
    assert setup.tr_pretrigger_samples == 100
    assert setup.tr_postduration_samples == 50
    assert {"channel"} == setup.extra.keys()


async def test_get_setup_since_v2_13(mock_objects):
    lw, reader, writer = mock_objects
    reader.readline.side_effect = [
        b"[ch1]\n",
        b"# dsp\n",
        b"input_range=1\n",  # adc_range -> input_range
        b"adc2uv=1.5625\n",
        b"filter=none - 300 kHz, order 8\n",
        b"# ae\n",
        b"enabled=1\n",
        b"cont=0\n",
        b"thr=100.0 uV\n",
        b"ddt=500.0 us\n",
        b"status_interval=100 ms\n",
        b"# tr\n",
        b"tr_enabled=1\n",
        b"tr_decimation=2\n",
        b"tr_pre_trig=100\n",
        b"tr_post_dur=50\n",
        b"\n",
        TimeoutError,
    ]
    setup = await lw.get_setup(1)
    writer.write.assert_called_with(b"get_setup @1\n")
    reader.readline.assert_awaited()

    assert setup.enabled is True
    assert setup.input_range == 1
    assert setup.adc_to_volts == 1.5625e-6
    assert setup.filter_highpass_hz is None
    assert setup.filter_lowpass_hz == 300e3
    assert setup.filter_order == 8
    assert setup.continuous_mode is False
    assert setup.threshold_volts == 100e-6
    assert setup.ddt_seconds == 500e-6
    assert setup.status_interval_seconds == 0.1
    assert setup.tr_enabled is True
    assert setup.tr_decimation == 2
    assert setup.tr_pretrigger_samples == 100
    assert setup.tr_postduration_samples == 50
    assert setup.extra == {}


@pytest.mark.parametrize(
    ("channel", "enabled", "command"),
    [
        (0, True, b"set_acq enabled 1 @0\n"),
        (1, False, b"set_acq enabled 0 @1\n"),
    ],
)
async def test_set_channel(mock_objects, channel, enabled, command):
    lw, _, writer = mock_objects
    await lw.set_channel(channel, enabled)
    writer.write.assert_called_with(command)


@pytest.mark.parametrize(
    ("channel", "index", "command"),
    [
        (0, 0, b"set_adc_range 0 @0\n"),
        (1, 1, b"set_adc_range 1 @1\n"),
        (2, 1, b"set_adc_range 1 @2\n"),
    ],
)
async def test_set_range_index(mock_objects, channel, index, command):
    lw, _, writer = mock_objects
    await lw.set_range_index(channel, index)
    writer.write.assert_called_with(command)


@pytest.mark.parametrize("channel", [-1, 3])
async def test_set_range_index_invalid_channel(mock_objects, channel):
    with pytest.raises(ValueError):
        await mock_objects.lw.set_range_index(channel, 0)


@pytest.mark.parametrize(
    ("channel", "value", "command"),
    [
        (0, 0.05, b"set_adc_range 0 @0\n"),
        (1, 5, b"set_adc_range 1 @1\n"),
        (2, 5, b"set_adc_range 1 @2\n"),
    ],
)
async def test_set_range(mock_objects, channel, value, command):
    lw, _, writer = mock_objects
    await lw.set_range(channel, value)
    writer.write.assert_called_with(command)


@pytest.mark.parametrize("value", [-1, 0, 11])
async def test_set_range_invalid_value(mock_objects, value):
    with pytest.raises(ValueError):
        await mock_objects.lw.set_range(0, value)


@pytest.mark.parametrize(
    ("channel", "enabled", "command"),
    [
        (0, True, b"set_acq cont 1 @0\n"),
        (2, False, b"set_acq cont 0 @2\n"),
    ],
)
async def test_set_continuous_mode(mock_objects, channel, enabled, command):
    lw, _, writer = mock_objects
    await lw.set_continuous_mode(channel, enabled)
    writer.write.assert_called_with(command)


@pytest.mark.parametrize(
    ("channel", "microseconds", "command"),
    [
        (0, 0, b"set_acq ddt 0 @0\n"),
        (1, 200, b"set_acq ddt 200 @1\n"),
    ],
)
async def test_set_ddt(mock_objects, channel, microseconds, command):
    lw, _, writer = mock_objects
    await lw.set_ddt(channel, microseconds)
    writer.write.assert_called_with(command)


@pytest.mark.parametrize(
    ("channel", "seconds", "command"),
    [
        (0, 0, b"set_acq status_interval 0 @0\n"),
        (1, 0.01, b"set_acq status_interval 10 @1\n"),
        (2, 10, b"set_acq status_interval 10000 @2\n"),
    ],
)
async def test_set_status_interval(mock_objects, channel, seconds, command):
    lw, _, writer = mock_objects
    await lw.set_status_interval(channel, seconds)
    writer.write.assert_called_with(command)


@pytest.mark.parametrize(
    ("channel", "enabled", "command"),
    [
        (0, True, b"set_acq tr_enabled 1 @0\n"),
        (1, False, b"set_acq tr_enabled 0 @1\n"),
    ],
)
async def test_set_tr_enabled(mock_objects, channel, enabled, command):
    lw, _, writer = mock_objects
    await lw.set_tr_enabled(channel, enabled)
    writer.write.assert_called_with(command)


@pytest.mark.parametrize(
    ("channel", "value", "command"),
    [
        (0, 1, b"set_acq tr_decimation 1 @0\n"),
        (1, 4, b"set_acq tr_decimation 4 @1\n"),
        (2, 8, b"set_acq tr_decimation 8 @2\n"),
        (0, 1000, b"set_acq tr_decimation 1000 @0\n"),
    ],
)
async def test_set_tr_decimation(mock_objects, channel, value, command):
    lw, _, writer = mock_objects
    await lw.set_tr_decimation(channel, value)
    writer.write.assert_called_with(command)


@pytest.mark.parametrize("channel", [-1, 3])
async def test_set_tr_decimation_invalid_channel(mock_objects, channel):
    with pytest.raises(ValueError):
        await mock_objects.lw.set_tr_decimation(channel, 1)


@pytest.mark.parametrize(
    ("channel", "highpass", "lowpass", "order", "command"),
    [
        (0, None, None, 4, b"set_filter none none 4 @0\n"),
        (1, 0, 0, 0, b"set_filter 0.0 0.0 0 @1\n"),
        (1, 10e3, 350e3, 8, b"set_filter 10.0 350.0 8 @1\n"),
        (2, 10e3, None, 8, b"set_filter 10.0 none 8 @2\n"),
    ],
)
async def test_set_filter(mock_objects, channel, highpass, lowpass, order, command):
    lw, _, writer = mock_objects
    await lw.set_filter(channel, highpass, lowpass, order)
    writer.write.assert_called_with(command)


@pytest.mark.parametrize("channel", [-1, 3])
async def test_set_filter_invalid_channel(mock_objects, channel):
    with pytest.raises(ValueError):
        await mock_objects.lw.set_filter(channel, 50e3, 300e3, 4)


@pytest.mark.parametrize(
    ("channel", "samples", "command"),
    [
        (0, 0, b"set_acq tr_pre_trig 0 @0\n"),
        (1, 200, b"set_acq tr_pre_trig 200 @1\n"),
    ],
)
async def test_set_tr_pretrigger(mock_objects, channel, samples, command):
    lw, _, writer = mock_objects
    await lw.set_tr_pretrigger(channel, samples)
    writer.write.assert_called_with(command)


@pytest.mark.parametrize(
    ("channel", "samples", "command"),
    [
        (0, 0, b"set_acq tr_post_dur 0 @0\n"),
        (2, 1000, b"set_acq tr_post_dur 1000 @2\n"),
    ],
)
async def test_set_tr_postduration(mock_objects, channel, samples, command):
    lw, _, writer = mock_objects
    await lw.set_tr_postduration(channel, samples)
    writer.write.assert_called_with(command)


@pytest.mark.parametrize(
    ("channel", "threshold", "command"),
    [
        (0, 0, b"set_acq thr 0 @0\n"),
        (2, 0.1, b"set_acq thr 0.1 @2\n"),
    ],
)
async def test_set_threshold(mock_objects, channel, threshold, command):
    lw, _, writer = mock_objects
    await lw.set_threshold(channel, threshold)
    writer.write.assert_called_with(command)


async def test_start_stop_acquisition(mock_objects):
    lw, _, writer = mock_objects
    await lw.start_acquisition()
    writer.write.assert_called_with(b"start_acq\n")
    await lw.stop_acquisition()
    writer.write.assert_called_with(b"stop_acq\n")


@pytest.mark.parametrize(
    ("channel", "interval", "count", "cycles", "command"),
    [
        (0, 1, 4, 1, b"start_pulsing 1 4 1 @0\n"),
        (1, 0.1, 0, 0, b"start_pulsing 0.1 0 0 @1\n"),
    ],
)
async def test_start_pulsing(mock_objects, channel, interval, count, cycles, command):
    lw, _, writer = mock_objects
    await lw.start_pulsing(channel, interval, count, cycles)
    writer.write.assert_called_with(command)


async def test_stop_pulsing(mock_objects):
    lw, _, writer = mock_objects
    await lw.stop_pulsing()
    writer.write.assert_called_with(b"stop_pulsing\n")


async def test_get_ae_data(mock_objects):
    lw, reader, writer = mock_objects
    reader.readline.side_effect = [
        b"S Ch=1 T=20000000 A=0 R=0 D=10000000 C=0 E=38788614 TRAI=0 flags=0\n",
        b"H Ch=2 T=43686000 A=31004 R=496 D=703 C=4 E=74860056830 TRAI=1 flags=0\n",
        b"\n",
        TimeoutError,
    ]

    ae_data = await lw.get_ae_data()
    writer.write.assert_called_with(b"get_ae_data\n")
    reader.readline.assert_awaited()

    assert len(ae_data) == 2

    adc_to_volts = ADC_TO_VOLTS[0]
    adc_to_eu = ADC_TO_EU[0]

    # status record
    s = ae_data[0]
    assert s.type_ == "S"
    assert s.channel == 1
    assert s.time == pytest.approx(20000000 / CLOCK)
    assert s.amplitude == 0
    assert s.rise_time == 0
    assert s.duration == pytest.approx(10000000 / CLOCK)
    assert s.counts == 0
    assert s.energy == pytest.approx(38788614 * adc_to_eu)
    assert s.trai == 0
    assert s.flags == 0

    # hit record
    h = ae_data[1]
    assert h.type_ == "H"
    assert h.channel == 2
    assert h.time == pytest.approx(43686000 / CLOCK)
    assert h.amplitude == pytest.approx(31004 * adc_to_volts)
    assert h.rise_time == pytest.approx(496 / CLOCK)
    assert h.duration == pytest.approx(703 / CLOCK)
    assert h.counts == 4
    assert h.energy == pytest.approx(74860056830 * adc_to_eu)
    assert h.trai == 1
    assert h.flags == 0


@pytest.mark.parametrize("raw", [False, True])
async def test_get_tr_data(mock_objects, raw):
    lw, reader, writer = mock_objects
    reader.readline.side_effect = [
        b"Ch=1 TRAI=1 T=43686000 NS=13\n",
        b"Ch=2 TRAI=2 T=43686983 NS=27\n",
        b"\n",
        TimeoutError,
    ]
    data = [np.arange(samples, dtype=np.int16) for samples in (13, 27)]
    binary_data = [arr.tobytes() for arr in data]

    reader.readexactly.side_effect = binary_data

    tr_data = await lw.get_tr_data(raw=raw)
    writer.write.assert_called_with(b"get_tr_data\n")

    adc_to_volts = ADC_TO_VOLTS[0]

    assert tr_data[0].channel == 1
    assert tr_data[0].trai == 1
    assert tr_data[0].time == pytest.approx(43686000 / CLOCK)
    assert tr_data[0].samples == 13
    assert tr_data[0].raw == raw
    if raw:
        assert_allclose(tr_data[0].data, data[0])
    else:
        assert_allclose(tr_data[0].data, data[0] * adc_to_volts)

    assert tr_data[1].channel == 2
    assert tr_data[1].trai == 2
    assert tr_data[1].time == pytest.approx(43686983 / CLOCK)
    assert tr_data[1].samples == 27
    assert tr_data[1].raw == raw
    if raw:
        assert_allclose(tr_data[1].data, data[1])
    else:
        assert_allclose(tr_data[1].data, data[1] * adc_to_volts)


async def test_get_tr_snapshot(mock_objects):
    lw, reader, writer = mock_objects
    reader.readline.side_effect = [
        b"Ch=1 NS=20 NB=40\n",
        b"Ch=2 NS=20 NB=40\n",
        b"\n",
        TimeoutError,
    ]
    data = [np.arange(samples, dtype=np.int16) for samples in (20, 20)]
    binary_data = [arr.tobytes() for arr in data]

    reader.readexactly.side_effect = binary_data

    tr_data = await lw.get_tr_snapshot(0, samples=20, pretrigger_samples=0, raw=True)
    writer.write.assert_called_with(b"get_tr_snapshot 20 @0\n")

    assert tr_data[0].channel == 1
    assert tr_data[0].trai == 0
    assert tr_data[0].time == 0
    assert tr_data[0].samples == 20
    assert tr_data[0].raw is True
    assert_allclose(tr_data[0].data, data[0])

    assert tr_data[1].channel == 2
    assert tr_data[1].trai == 0
    assert tr_data[1].time == 0
    assert tr_data[1].samples == 20
    assert tr_data[1].raw is True
    assert_allclose(tr_data[1].data, data[1])
