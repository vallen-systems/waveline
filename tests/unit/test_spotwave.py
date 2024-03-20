from datetime import datetime
from typing import NamedTuple
from unittest.mock import Mock, patch

import numpy as np
import pytest
from freezegun import freeze_time
from numpy.testing import assert_allclose
from serial import Serial, SerialException
from waveline import SpotWave
from waveline.datatypes import Info

CLOCK = 2e6
ADC_TO_VOLTS = 1.74e-6
ADC_TO_EU = ADC_TO_VOLTS**2 * 1e14 / CLOCK


class MockedObjects(NamedTuple):
    sw: SpotWave
    serial: Serial


@pytest.fixture(autouse=True, name="mock_objects")
def mock_spotwave():
    serial = Mock(spec=Serial)
    serial.is_open = False

    with patch.object(SpotWave, "get_info") as mock_get_info:
        mock_get_info.return_value = Info(
            hardware_id="002E004B3139511638303932",
            firmware_version="00.2C",
            channel_count=1,
            input_range=["94 dBAE"],
            adc_to_volts=[ADC_TO_VOLTS],
            extra={},
        )
        sw = SpotWave(serial)
    return MockedObjects(sw, serial)


def test_init_serial(mock_objects):
    sw, serial = mock_objects
    assert sw._ser == serial  # pylint: disable=protected-access
    serial.open.assert_called()


def test_init_port():
    with pytest.raises(SerialException):
        SpotWave("invalid_port_id")


def test_init_invalid_type():
    with pytest.raises(ValueError):
        SpotWave(123)


def test_get_info(mock_objects):
    sw, serial = mock_objects
    serial.readline.side_effect = [
        b"hw_id=002E004B3139511638303932\n",  # included in firmware version 00.2E
        b"fw_version=00.2C\n",
        b"type=spotWave\n",
        b"model=201\n",
        b"adc2uv=1.74 uV\n",
        b"input_range=94 dBAE\n",
        b"input_resistance=16 kOhm\n",
        b"input_capacity=12 pF\n",
        b"max_samplerate=2 MHz\n",
        b"analog_bandwidth=20-500 kHz\n",
        b"cct_voltage=3.3 V\n",
        b"flash_memory=64 MB\n",
        b"serial_number=0007\n",
        b"pcb_vid=200505-06-0123\n",
        b"verification=2021-01-01 06:41:09.54\n",
        b"\n",
        b"",
    ]
    info = sw.get_info()
    serial.write.assert_called_with(b"get_info\n")

    assert info.hardware_id == "002E004B3139511638303932"
    assert info.firmware_version == "00.2C"
    assert info.channel_count == 1
    assert info.input_range == ["94 dBAE"]
    assert info.adc_to_volts == [1.74e-6]
    assert {
        "type",
        "model",
        "input_resistance",
        "input_capacity",
        "max_samplerate",
        "analog_bandwidth",
        "cct_voltage",
        "flash_memory",
        "serial_number",
        "pcb_vid",
        "verification",
    } == info.extra.keys()

    # empty response
    serial.readline.side_effects = [b""]
    with pytest.raises(RuntimeError):
        sw.get_info()


def test_get_status(mock_objects):
    sw, serial = mock_objects
    serial.readline.side_effect = [
        b"temp=24 \xc2\xb0C\n",
        b"recording=0\n",
        b"logging=0\n",
        b"pulsing=1\n",
        b"usb_speed=high\n",
        b"log_data_usage=13 sets (0.12 %)\n",
        b"date=2020-12-17 15:11:42.17\n",
        b"",
    ]
    status = sw.get_status()
    serial.write.assert_called_with(b"get_status\n")

    assert status.temperature == 24
    assert status.recording is False
    assert status.pulsing is True
    assert {"logging", "usb_speed", "log_data_usage", "date"} == status.extra.keys()

    # empty response
    serial.readline.side_effect = [b""]
    with pytest.raises(RuntimeError):
        sw.get_status()


def test_get_setup(mock_objects):
    sw, serial = mock_objects
    response = [
        b"recording=1\n",
        b"logging=0\n",
        b"adc2uv=1.74\n",
        b"filter=10.5 - 350 kHz, order 4\n",
        b"cont=0\n",
        b"thr=3162.5 uV\n",
        b"ddt=250  us\n",
        b"dummy line without value",  # modified on puporse
        b" status_interval = 1000 ms\n",  # modified on purpose
        b"tr_enabled=1\n",
        b"tr_decimation=2\n",
        b"tr_pre_trig=100\n",
        b"tr_post_dur=100\n",
        b"\n",
        b"",
    ]
    serial.readline.side_effect = response
    setup = sw.get_setup()
    serial.write.assert_called_with(b"get_setup\n")
    assert setup.enabled is True
    assert setup.input_range == 0
    assert setup.adc_to_volts == 1.74e-6
    assert setup.filter_highpass_hz == 10.5e3
    assert setup.filter_lowpass_hz == 350e3
    assert setup.filter_order == 4
    assert setup.continuous_mode is False
    assert setup.threshold_volts == 3162.5e-6
    assert setup.ddt_seconds == 250e-6
    assert setup.status_interval_seconds == 1
    assert setup.tr_enabled is True
    assert setup.tr_decimation == 2
    assert setup.tr_pretrigger_samples == 100
    assert setup.tr_postduration_samples == 100
    assert {"recording", "logging"} == setup.extra.keys()

    # test special filter outputs
    response[3] = b"filter=none-350 kHz, order 4\n"
    serial.readline.side_effect = response
    setup = sw.get_setup()
    assert setup.filter_highpass_hz is None
    assert setup.filter_lowpass_hz == 350_000
    assert setup.filter_order == 4

    response[3] = b"filter=10.5-none kHz, order 4\n"
    serial.readline.side_effect = response
    setup = sw.get_setup()
    assert setup.filter_highpass_hz == 10_500
    assert setup.filter_lowpass_hz is None
    assert setup.filter_order == 4

    response[3] = b"filter=none-none kHz, order 0\n"
    serial.readline.side_effect = response
    setup = sw.get_setup()
    assert setup.filter_highpass_hz is None
    assert setup.filter_lowpass_hz is None
    assert setup.filter_order == 0

    # empty response
    serial.readline.side_effect = [b""]
    with pytest.raises(RuntimeError):
        sw.get_setup()


@freeze_time("2022-11-11")
def test_commands_without_response(mock_objects):
    sw, serial = mock_objects

    def assert_write(expected):
        serial.write.assert_called_with(expected)

    sw.identify()
    assert_write(b"identify\n")

    sw.set_continuous_mode(True)
    assert_write(b"set_acq cont 1\n")

    sw.set_logging_mode(True)
    assert_write(b"set_data_log enabled 1\n")
    sw.set_logging_mode(False)
    assert_write(b"set_data_log enabled 0\n")

    sw.set_ddt(400)
    assert_write(b"set_acq ddt 400\n")

    sw.set_status_interval(2.2)
    assert_write(b"set_acq status_interval 2200\n")

    sw.set_tr_enabled(True)
    assert_write(b"set_acq tr_enabled 1\n")

    sw.set_tr_decimation(10)
    assert_write(b"set_acq tr_decimation 10\n")

    sw.set_tr_pretrigger(200)
    assert_write(b"set_acq tr_pre_trig 200\n")

    sw.set_tr_postduration(0)
    assert_write(b"set_acq tr_post_dur 0\n")

    sw.set_cct(0.1)
    assert_write(b"set_cct interval 0.1\n")
    sw.set_cct(-1)
    assert_write(b"set_cct interval -1\n")

    sw.set_filter(highpass=100_000, lowpass=300_000, order=6)
    assert_write(b"set_filter 100.0 300.0 6\n")

    sw.set_datetime()  # datetime.now() -> frozen time
    assert_write(b"set_datetime 2022-11-11 00:00:00\n")

    sw.set_datetime(datetime(2020, 12, 16, 17, 55, 13))
    assert_write(b"set_datetime 2020-12-16 17:55:13\n")

    sw.set_threshold(100)
    assert_write(b"set_acq thr 100\n")

    sw.start_acquisition()
    assert_write(b"start_acq\n")

    sw.stop_acquisition()
    assert_write(b"stop_acq\n")


def test_get_ae_data(mock_objects):
    sw, serial = mock_objects
    serial.readline.side_effect = [
        b"S temp=27 T = 2010240 A=21 R=502689 D=2000000 C=0 E=38849818 TRAI=0 flags=0\n",
        b"H temp=27 T=3044759 A=3557 R=24 D=819 C=31 E=518280026 TRAI=1 flags=0\n",
        b"\n",
    ]
    ae_data = sw.get_ae_data()

    # status record
    s = ae_data[0]
    assert s.type_ == "S"
    assert s.channel == 1
    assert s.time == pytest.approx(2010240 / CLOCK)
    assert s.amplitude == pytest.approx(21 * ADC_TO_VOLTS)
    assert s.rise_time == pytest.approx(502689 / CLOCK)
    assert s.duration == pytest.approx(2000000 / CLOCK)
    assert s.counts == 0
    assert s.energy == pytest.approx(38849818 * ADC_TO_EU)
    assert s.trai == 0
    assert s.flags == 0

    # hit record
    h = ae_data[1]
    assert h.type_ == "H"
    assert h.channel == 1
    assert h.time == pytest.approx(3044759 / CLOCK)
    assert h.amplitude == pytest.approx(3557 * ADC_TO_VOLTS)
    assert h.rise_time == pytest.approx(24 / CLOCK)
    assert h.duration == pytest.approx(819 / CLOCK)
    assert h.counts == 31
    assert h.energy == pytest.approx(518280026 * ADC_TO_EU)
    assert h.trai == 1
    assert h.flags == 0


@pytest.mark.parametrize("raw", [False, True])
def test_get_tr_data(mock_objects, raw):
    sw, serial = mock_objects

    lines = [
        b"TRAI = 1 T=43686000 NS=13\n",
        b"TRAI=2 T=43686983 NS=27\n",
        b"\n",
    ]
    data = [np.arange(samples, dtype=np.int16) for samples in (13, 27)]
    binary_data = [arr.tobytes() for arr in data]

    serial.readline.side_effect = lines
    serial.read.side_effect = binary_data

    tr_data = sw.get_tr_data(raw=raw)
    serial.write.assert_called_with(b"get_tr_data\n")

    assert tr_data[0].channel == 1
    assert tr_data[0].trai == 1
    assert tr_data[0].time == pytest.approx(43686000 / CLOCK)
    assert tr_data[0].samples == 13
    assert tr_data[0].raw == raw
    if raw:
        assert_allclose(tr_data[0].data, data[0])
    else:
        assert_allclose(tr_data[0].data, data[0] * ADC_TO_VOLTS)

    assert tr_data[1].channel == 1
    assert tr_data[1].trai == 2
    assert tr_data[1].time == pytest.approx(43686983 / CLOCK)
    assert tr_data[1].samples == 27
    assert tr_data[1].raw == raw
    if raw:
        assert_allclose(tr_data[1].data, data[1])
    else:
        assert_allclose(tr_data[1].data, data[1] * ADC_TO_VOLTS)


@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize(
    "samples",
    [
        32,
        128,
        65536,
    ],
)
def test_get_data(mock_objects, samples, raw):
    sw, serial = mock_objects

    mock_data = (2**15 * np.random.randn(samples)).astype(np.int16)
    serial.readline.return_value = f"NS={samples}\n".encode()
    serial.read.return_value = mock_data.tobytes()

    data = sw.get_data(samples, raw=raw)
    serial.write.assert_called_with(f"get_tr_snapshot {samples}\n".encode())
    if raw:
        assert_allclose(data, mock_data)
    else:
        assert_allclose(data, mock_data * ADC_TO_VOLTS)


@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize(
    "samples",
    [
        32,
        128,
        65536,
    ],
)
def test_get_tr_snapshot(mock_objects, samples, raw):
    sw, serial = mock_objects

    mock_data = (2**15 * np.random.randn(samples)).astype(np.int16)
    serial.readline.return_value = f"NS={samples}\n".encode()
    serial.read.return_value = mock_data.tobytes()

    tr_record = sw.get_tr_snapshot(samples, raw=raw)
    serial.write.assert_called_with(f"get_tr_snapshot {samples}\n".encode())
    assert tr_record.channel == 1
    assert tr_record.trai == 0
    assert tr_record.time == 0
    assert tr_record.samples == samples
    assert tr_record.raw == raw
    if raw:
        assert_allclose(tr_record.data, mock_data)
    else:
        assert_allclose(tr_record.data, mock_data * ADC_TO_VOLTS)
