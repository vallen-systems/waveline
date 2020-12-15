from datetime import datetime
from textwrap import dedent
from unittest.mock import Mock, patch

import numpy as np
import pytest
from freezegun import freeze_time
from serial import Serial, SerialException

from waveline import SpotWave
from waveline.spotwave import Setup

RESPONSE_GET_AE_DATA = b"""
2
S temp=27 T=2010240 A=21 R=502689 D=2000000 C=0 E=38849818 TRAI=0 flags=0
H temp=27 T=3044759 A=3557 R=24 D=819 C=31 E=518280026 TRAI=1 flags=0
""".lstrip()

RESPONSE_GET_TR_DATA = """
TRAI=1 T=43686000 NS=768
{tra1}
TRAI=2 T=43686983 NS=692
{tra2}
""".format(
    tra1="\n".join([str(x) for x in range(768)]),
    tra2="\n".join([str(x) for x in range(692)]),
).encode().lstrip()


@pytest.fixture()
def serial_mock():
    serial_mock = Mock(spec=Serial)
    serial_mock.is_open = False
    return serial_mock


def test_init_serial(serial_mock):
    sw = SpotWave(serial_mock)
    assert sw._ser == serial_mock
    serial_mock.open.assert_called()


def test_init_port():
    with pytest.raises(SerialException):
        SpotWave("invalid_port_id")


def test_init_invalid_type():
    with pytest.raises(ValueError):
        SpotWave(123)


def test_get_setup(serial_mock):
    sw = SpotWave(serial_mock)

    response = [
        b"acq_enabled=1\n",
        b"log_enabled=0\n",
        b"adc2uv=1.74\n",
        b"cct=-0.5 s\n",
        b"dig.filter:  38-350 kHz, order=4, stages=4\n",
        b"cont=0\n",
        b"thr=3162.5 uV\n",
        b"ddt=250  us\n",
        b"status_interval=1000 ms\n",
        b"tr_enabled=1\n",
        b"tr_decimation=2\n",
        b"tr_pre_trig=100\n",
        b"tr_post_dur=100\n",
        b"tr_max_samples=2097152\n",
    ]
    serial_mock.readlines.return_value = response
    setup = sw.get_setup()
    serial_mock.write.assert_called_with(b"get_setup\n")
    assert setup == Setup(
        acq_enabled=True,
        cont_enabled=False,
        log_enabled=False,
        adc_to_volts=1.74e-6,
        threshold_volts=3162.5e-6,
        ddt_seconds=250e-6,
        status_interval_seconds=1,
        filter_highpass_hz=38e3,
        filter_lowpass_hz=350e3,
        filter_order=4,
        tr_enabled=1,
        tr_decimation=2,
        tr_pretrigger_samples=100,
        tr_postduration_samples=100,
        cct_seconds=-0.5,
    )

    # test special filter outputs
    response[4] = b"dig.filter: none"
    serial_mock.readlines.return_value = response
    setup = sw.get_setup()
    assert setup.filter_highpass_hz == 0
    assert setup.filter_lowpass_hz == 1_000_000
    assert setup.filter_order == 0

    response[4] = b"dig.filter:  10-max kHz, order=4, stages=2"
    serial_mock.readlines.return_value = response
    setup = sw.get_setup()
    assert setup.filter_highpass_hz == 10_000
    assert setup.filter_lowpass_hz == 1_000_000
    assert setup.filter_order == 4

    # empty response
    serial_mock.readlines.return_value = []
    with pytest.raises(RuntimeError):
        sw.get_setup()


def test_get_status(serial_mock):
    sw = SpotWave(serial_mock)

    response = [
        b"fw_version=00.1B\n",
        b"dev_id=0019003A3438511539373231\n",
        b"hw_rev=0\n",
        b"chip_rev=V (0x2003)\n",
        b"temp=24\n",
        b"recording=0\n",
        b"logging=0\n",
        b"data size=0\n",
        b"date=2020-12-17 15:11:42.17\n",
        b"shutdown time=2020-12-17 11:14:23.40\n",
        b"adc2uv=1.72\n",
        b"cct=0\n",
        b"dig.filter:  20-500 kHz, order=4, stages=4\n",
        b"taskMain       	74159		<1%\n",
        b"IDLE           	188137815		51%\n",
        b"taskADC        	177045212		48%\n",
        b"Tmr Svc        	0		<1%\n",
    ]
    serial_mock.readlines.return_value = response
    status = sw.get_status()
    serial_mock.write.assert_called_with(b"get_status\n")

    assert status.device_id == "0019003A3438511539373231"
    assert status.firmware_version == "00.1B"
    assert status.temperature == 24
    assert status.data_size == 0
    assert status.datetime == datetime(2020, 12, 17, 15, 11, 42, 170_000)

    # empty response
    serial_mock.readlines.return_value = []
    with pytest.raises(RuntimeError):
        sw.get_status()


@freeze_time("2022-11-11")
def test_commands_without_response(serial_mock):
    sw = SpotWave(serial_mock)

    def assert_write(expected):
        serial_mock.write.assert_called_with(expected)

    sw.set_continuous_mode(True)
    assert_write(b"set_acq cont 1\n")

    sw.set_ddt(400)
    assert_write(b"set_acq ddt 400\n")

    sw.set_status_interval(2000)
    assert_write(b"set_acq status_interval 2000\n")

    sw.set_tr_enabled(True)
    assert_write(b"set_acq tr_enabled 1\n")

    sw.set_tr_decimation(10)
    assert_write(b"set_acq tr_decimation 10\n")

    sw.set_tr_pretrigger(200)
    assert_write(b"set_acq tr_pre_trig 200\n")

    sw.set_tr_postduration(0)
    assert_write(b"set_acq tr_post_dur 0\n")

    sw.set_cct(-0.1)
    assert_write(b"set_cct -0.1\n")

    sw.set_filter(highpass=100, lowpass=300, order=6)
    assert_write(b"set_filter 100 300 6\n")

    sw.set_datetime(datetime(2020, 12, 16, 17, 55, 13))
    assert_write(b"set_datetime 2020-12-16 17:55:13\n")

    sw.set_datetime()  # datetime.now() -> frozen time
    assert_write(b"set_datetime 2022-11-11 00:00:00\n")

    sw.set_threshold(100)
    assert_write(b"set_acq thr 100\n")

    sw.start_acquisition()
    assert_write(b"set_acq enabled 1\n")

    sw.stop_acquisition()
    assert_write(b"set_acq enabled 0\n")
