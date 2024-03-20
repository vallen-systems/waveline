"""
Module for spotWave device.

All device-related functions are exposed by the `SpotWave` class.
"""

import logging
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator, List, Optional, Union
from warnings import warn

import numpy as np
from serial import EIGHTBITS, Serial
from serial.tools import list_ports

from ._common import (
    _check_firmware_version,
    _parse_ae_headerline,
    _parse_get_info_output,
    _parse_get_setup_output,
    _parse_get_status_output,
    _parse_tr_headerline,
)
from .datatypes import AERecord, Info, Setup, Status, TRRecord

logger = logging.getLogger(__name__)


class SpotWave:
    """
    Interface for spotWave devices.

    The spotWave device is connected via USB and exposes a virtual serial port for communication.
    """

    VENDOR_ID = 8849  #: USB vendor id of Vallen Systeme GmbH
    PRODUCT_ID = 272  #: USB product id of SpotWave device
    CLOCK = 2_000_000  #: Internal clock in Hz
    # TICKS_TO_SEC = 1 / CLOCK  # precision lost...?
    _MIN_FIRMWARE_VERSION = "00.25"

    def __init__(self, port: Union[str, Serial]):
        """
        Initialize device.

        Args:
            port: Either the serial port id (e.g. "COM6") or a `serial.Serial` port instance.
                Use the method `discover` to get a list of ports with connected spotWave devices.
        Returns:
            Instance of `SpotWave`

        Example:
            There are two ways constructing and using the `SpotWave` class:

            1.  Without context manager and manually calling the `close` method afterwards:

                >>> sw = waveline.SpotWave("COM6")
                >>> print(sw.get_setup())
                >>> ...
                >>> sw.close()

            2.  Using the context manager:

                >>> with waveline.SpotWave("COM6") as sw:
                >>>     print(sw.get_setup())
                >>>     ...
        """
        if isinstance(port, str):
            self._ser = Serial(port=port)
        elif isinstance(port, Serial):
            self._ser = port
        else:
            raise ValueError("Either pass a port id or a Serial instance")

        self._ser.baudrate = 115200  # doesn't matter
        self._ser.bytesize = EIGHTBITS
        self._ser.timeout = 1  # seconds
        self._ser.exclusive = True

        self.connect()
        info = self.get_info()
        _check_firmware_version(info.firmware_version, self._MIN_FIRMWARE_VERSION, base=16)
        self._adc_to_volts = info.adc_to_volts[0]  # TODO: support for multi-range devices

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def connect(self):
        """
        Open serial connection to the device.

        The `connect` method is automatically called in the constructor.
        You only need to call the method to reopen the connection after calling `close`.
        """
        if not self.connected:
            self._ser.open()

    def close(self):
        """Close serial connection to the device."""
        if not hasattr(self, "_ser"):
            return
        if self.connected:
            self._ser.close()

    @property
    def connected(self) -> bool:
        """Check if the connection to the device is open."""
        return self._ser.is_open

    @contextmanager
    def _timeout_context(self, timeout_seconds: float):
        """Temporary set serial read/write timeout within context."""
        old_timeout = self._ser.timeout
        self._ser.timeout = timeout_seconds
        yield None
        self._ser.timeout = old_timeout

    @classmethod
    def discover(cls) -> List[str]:
        """
        Discover connected spotWave devices.

        Returns:
            List of port names
        """
        ports = list_ports.comports()
        ports_spotwave = filter(
            lambda p: (p.vid == cls.VENDOR_ID) & (p.pid == cls.PRODUCT_ID),
            ports,
        )
        return [port.device for port in ports_spotwave]

    def identify(self):
        """
        Blink LED to identify device.

        Note:
            Available since firmware version 00.2D.
        """
        self._send_command("identify")

    def _readlines(self, return_emptyline: bool = True):
        """Read lines using custom timeout."""
        lines: List[bytes] = []
        # long timeout (1000 ms) for first line, then short timeouts (100 ms)
        with self._timeout_context(1.0):
            while True:
                if lines:
                    self._ser.timeout = 0.1
                line = self._ser.readline()
                if line == b"":  # -> timeout
                    break
                lines.append(line)
                if return_emptyline and line == b"\n":
                    break
        return lines

    def clear_buffer(self):
        """Clear input and output buffer."""
        self._readlines()
        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()

    def _send_command(self, command: str):
        command_bytes = command.encode("utf-8") + b"\n"  # str -> bytes
        logger.debug("Send command: %a", command_bytes)
        self._ser.write(command_bytes)

    def get_info(self) -> Info:
        """
        Get device information.

        Returns:
            Dataclass with device information
        """
        self._send_command("get_info")
        lines = self._readlines()
        if not lines:
            raise RuntimeError("Could not get device information")

        info = _parse_get_info_output(lines)
        info.channel_count = 1
        assert len(info.input_range) == len(info.adc_to_volts)
        return info

    def get_setup(self) -> Setup:
        """
        Get setup.

        Returns:
            Dataclass with setup information
        """
        self._send_command("get_setup")
        lines = self._readlines()
        if not lines:
            raise RuntimeError("Could not get setup")

        setup = _parse_get_setup_output(lines)
        setup.enabled = True  # channel is always enabled
        return setup

    def get_status(self) -> Status:
        """
        Get status.

        Returns:
            Dataclass with status information
        """
        self._send_command("get_status")
        lines = self._readlines()
        if not lines:
            raise RuntimeError("Could not get status")

        return _parse_get_status_output(lines)

    def set_continuous_mode(self, enabled: bool):
        """
        Enable/disable continuous mode.

        Threshold will be ignored in continous mode.
        The length of the records is determined by `ddt` with `set_ddt`.

        Note:
            The parameters for continuous mode with transient recording enabled (`set_tr_enabled`)
            have to be chosen with care - mainly the decimation factor (`set_tr_decimation`) and
            `ddt` (`set_ddt`). The internal buffer of the device can store up to ~200.000 samples.

            If the buffer is full, data records are lost.
            Small latencies in data polling can cause overflows and therefore data loss.
            One record should not exceed half the buffer size (~100.000 samples).
            25% of the buffer size (~50.000 samples) is a good starting point.
            The number of samples in a record is determined by `ddt` and the decimation factor `d`:
            :math:`n = ddt_{\\mu s} \\cdot f_s / d = ddt_{\\mu s} \\cdot 2 / d`
            :math:`\\implies ddt_{\\mu s} \\approx 50.000 \\cdot d / 2`

            On the other hand, if the number of samples is small, more hits are generated and the
            CPU load increases.

        Args:
            enabled: Set to `True` to enable continuous mode
        """
        self._send_command(f"set_acq cont {int(enabled)}")

    def set_ddt(self, microseconds: int):
        """
        Set duration discrimination time (DDT).

        Args:
            microseconds: DDT in µs
        """
        self._send_command(f"set_acq ddt {int(microseconds)}")

    def set_status_interval(self, seconds: int):
        """
        Set status interval.

        Args:
            seconds: Status interval in s
        """
        self._send_command(f"set_acq status_interval {int(seconds * 1e3)}")

    def set_tr_enabled(self, enabled: bool):
        """
        Enable/disable recording of transient data.

        Args:
            enabled: Set to `True` to enable transient data
        """
        self._send_command(f"set_acq tr_enabled {int(enabled)}")

    def set_tr_decimation(self, factor: int):
        """
        Set decimation factor of transient data.

        The sampling rate of transient data will be 2 MHz / `factor`.

        Args:
            factor: Decimation factor
        """
        self._send_command(f"set_acq tr_decimation {int(factor)}")

    def set_tr_pretrigger(self, samples: int):
        """
        Set pre-trigger samples for transient data.

        Args:
            samples: Pre-trigger samples
        """
        self._send_command(f"set_acq tr_pre_trig {int(samples)}")

    def set_tr_postduration(self, samples: int):
        """
        Set post-duration samples for transient data.

        Args:
            samples: Post-duration samples
        """
        self._send_command(f"set_acq tr_post_dur {int(samples)}")

    def set_cct(self, interval_seconds: float):
        """
        Set coupling check ransmitter (CCT) / pulser interval.

        The pulser amplitude is 3.3 V.

        Args:
            interval_seconds: Pulser interval in seconds. If < 0, the pulse is synchronized with the
                              first sample of the `get_tr_snapshot` command.
        """
        self._send_command(f"set_cct interval {interval_seconds}")

    def set_filter(
        self,
        highpass: Optional[float] = None,
        lowpass: Optional[float] = None,
        order: int = 4,
    ):
        """
        Set IIR filter frequencies and order.

        Args:
            highpass: Highpass frequency in Hz (`None` to disable highpass filter)
            lowpass: Lowpass frequency in Hz (`None` to disable lowpass filter)
            order: Filter order
        """

        def khz_or_none(freq: Optional[float]):
            return freq / 1e3 if freq is not None else "none"

        self._send_command(
            f"set_filter {khz_or_none(highpass)} {khz_or_none(lowpass)} {int(order)}"
        )

    def set_datetime(self, timestamp: Optional[datetime] = None):
        """
        Set current date and time.

        Args:
            timestamp: `datetime.datetime` object, current time if `None`
        """
        if not timestamp:
            timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        self._send_command(f"set_datetime {timestamp_str}")

    def set_threshold(self, microvolts: float):
        """
        Set threshold for hit-based acquisition.

        Args:
            microvolts: Threshold in µV
        """
        self._send_command(f"set_acq thr {microvolts}")

    def set_logging_mode(self, enabled: bool):
        """
        Enable/disable data log mode.

        Args:
            enabled: Set to `True` to enable logging mode
        """
        self._send_command(f"set_data_log enabled {int(enabled)}")

    def start_acquisition(self):
        """Start acquisition."""
        logger.info("Start acquisition")
        self._send_command("start_acq")

    def stop_acquisition(self):
        """Stop acquisition."""
        logger.info("Stop acquisition")
        self._send_command("stop_acq")

    def start_pulsing(self, interval: float = 1, count: int = 0):
        """
        Start pulsing.

        The pulser amplitude is 3.3 V.
        The number of pulses should be even, because pulses are generated by a square-wave signal
        (between LOW and HIGH) and the pulse signal should end LOW.

        Args:
            interval: Interval between pulses in seconds
            count: Number of pulses per channel (should be even), 0 for infinite pulses
        """
        self._send_command(f"start_pulsing {interval} {count}")

    def stop_pulsing(self):
        """Stop pulsing."""
        self._send_command("stop_pulsing")

    def _read_ae_data(self) -> List[AERecord]:
        records = []
        while True:
            line = self._ser.readline()
            if line == b"\n":  # last line is an empty new line
                break

            record = _parse_ae_headerline(
                line, self.CLOCK, lambda _: self._adc_to_volts, default_channel=1
            )
            if record is not None:
                records.append(record)
        return records

    def get_ae_data(self) -> List[AERecord]:
        """
        Get AE data records.

        Returns:
            List of AE data records (either status or hit data)
        """
        self._send_command("get_ae_data")
        return self._read_ae_data()

    def _read_tr_data(self, raw: bool, records_expected: Optional[int] = None) -> List[TRRecord]:
        records = []
        while True:
            headerline = self._ser.readline()
            if headerline == b"\n":  # last line is an empty new line
                break

            record = _parse_tr_headerline(headerline, self.CLOCK, default_channel=1)
            record.data = np.frombuffer(self._ser.read(2 * record.samples), dtype=np.int16)
            record.raw = raw
            assert len(record.data) == record.samples
            if not raw:
                record.data = np.multiply(record.data, self._adc_to_volts, dtype=np.float32)
            records.append(record)
            if records_expected is not None and len(records) >= records_expected:
                break
        return records

    def get_tr_data(self, raw: bool = False) -> List[TRRecord]:
        """
        Get transient data records.

        Args:
            raw: Return TR amplitudes as ADC values if `True`, skip conversion to volts

        Returns:
            List of transient data records
        """
        self._send_command("get_tr_data")
        return self._read_tr_data(raw)

    def get_tr_snapshot(self, samples: int, raw: bool = False) -> TRRecord:
        """
        Read snapshot of transient data.

        The recording starts with the execution of the command.
        The trai and time of the returned records are always `0`.

        Args:
            samples: Number of samples to read
            raw: Return ADC values if `True`, skip conversion to volts

        Returns:
            Transient data record
        """
        self._send_command(f"get_tr_snapshot {int(samples)}")
        return self._read_tr_data(raw, records_expected=1)[0]

    def get_data(self, samples: int, raw: bool = False) -> np.ndarray:
        """
        Read snapshot of transient data.

        The recording starts with the execution of the command.

        Deprecated: Please us the `get_tr_snapshot` method instead.

        Args:
            samples: Number of samples to read
            raw: Return ADC values if `True`, skip conversion to volts

        Returns:
            Array with amplitudes in volts (or ADC values if `raw` is `True`)
        """
        warn(
            (
                "This method is deprecated and will be removed in the future. "
                "Please use the get_tr_snapshot method instead."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_tr_snapshot(samples, raw).data

    def acquire(
        self,
        raw: bool = False,
        poll_interval_seconds: float = 0.01,
    ) -> Iterator[Union[AERecord, TRRecord]]:
        """
        High-level method to continuously acquire data.

        Args:
            raw: Return TR amplitudes as ADC values if `True`, skip conversion to volts
            poll_interval_seconds: Pause between data polls in seconds

        Yields:
            AE and TR data records

        Example:
            >>> with waveline.SpotWave("COM6") as sw:
            >>>     # apply settings
            >>>     sw.set_ddt(400)
            >>>     for record in sw.stream():
            >>>         # do something with the data depending on the type
            >>>         if isinstance(record, waveline.AERecord):
            >>>             ...
            >>>         if isinstance(record, waveline.TRRecord):
            >>>             ...
        """
        self.start_acquisition()
        try:
            while True:
                t = time.monotonic()
                yield from self.get_ae_data()
                yield from self.get_tr_data(raw=raw)
                t = time.monotonic() - t
                # avoid brute load
                if t < poll_interval_seconds:
                    time.sleep(poll_interval_seconds)
        finally:
            self.stop_acquisition()

    def stream(self, *args, **kwargs):
        """
        Alias for `SpotWave.acquire` method.

        Deprecated: Please us the `acquire` method instead.
        """
        warn(
            (
                "This method is deprecated and will be removed in the future. "
                "Please use the acquire method instead."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        return self.acquire(*args, **kwargs)

    def get_data_log(self) -> List[AERecord]:
        """
        Get logged AE data records data from internal memory

        Returns:
            List of AE data records (either status or hit data)
        """
        self._send_command("get_data_log")
        return self._read_ae_data()

    def clear_data_log(self):
        """Clear logged data from internal memory."""
        self._send_command("clear_data_log")
