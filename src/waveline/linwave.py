"""
Module for linWave device.

All device-related functions are exposed by the `LinWave` class.
"""

import asyncio
import logging
import socket
import time
from dataclasses import dataclass, replace
from functools import wraps
from typing import AsyncIterator, ClassVar, Dict, List, Optional, Set, Tuple, Union
from warnings import warn

import numpy as np

from ._common import (
    KV_PATTERN,
    as_float,
    as_int,
    dict_get_first,
    multiline_output_to_dict,
    parse_array,
    parse_filter_setup_line,
)
from .datatypes import AERecord, TRRecord

logger = logging.getLogger(__name__)


@dataclass
class Info:
    """Device information."""

    hardware_id: str  #: Unique hardware id (since firmware version 2.13)
    firmware_version: str  #: Firmware version
    fpga_version: str  #: FPGA version
    channel_count: int  #: Number of channels
    input_range: List[str]  #: List of selectable input ranges
    max_sample_rate: float  #: Max sampling rate
    adc_to_volts: List[float]  #: Conversion factors from ADC values to V for both ranges


@dataclass
class Status:
    """Status information."""

    temperature: float  #: Device temperature in °C
    buffer_size: int  #: Buffer size in bytes


@dataclass
class Setup:
    """Setup."""

    adc_range_volts: float  #: ADC input range in volts
    adc_to_volts: float  #: Conversion factor from ADC values to volts
    filter_highpass_hz: Optional[float]  #: Highpass frequency in Hz
    filter_lowpass_hz: Optional[float]  #: Lowpass frequency in Hz
    filter_order: int  #: Filter order
    enabled: bool  #: Flag if channel is enabled
    continuous_mode: bool  #: Flag if continuous mode is enabled
    threshold_volts: float  #: Threshold for hit-based acquisition in volts
    ddt_seconds: float  #: Duration discrimination time (DDT) in seconds
    status_interval_seconds: float  #: Status interval in seconds
    tr_enabled: bool  #: Flag in transient data recording is enabled
    tr_decimation: int  #: Decimation factor for transient data
    tr_pretrigger_samples: int  #: Pre-trigger samples for transient data
    tr_postduration_samples: int  #: Post-duration samples for transient data


@dataclass
class _ChannelSettings:
    """Channel settings."""

    range_index: int  #: Input range in volts
    decimation: int  #: Decimation factor


def _require_connected(func):
    def check(obj: "LinWave"):
        if not obj.connected:
            raise ValueError("Device not connected")

    @wraps(func)
    async def async_wrapper(self: "LinWave", *args, **kwargs):
        check(self)
        return await func(self, *args, **kwargs)

    @wraps(func)
    def sync_wrapper(self: "LinWave", *args, **kwargs):
        check(self)
        return func(self, *args, **kwargs)

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


def _channel_str(channel: int) -> str:
    if channel == 0:
        return "all channels"
    return f"channel {channel:d}"


def _adc_to_eu(adc_to_volts: List[float], samplerate: float) -> List[float]:
    return [factor**2 * 1e14 / samplerate for factor in adc_to_volts]


def _check_firmware_version(firmware_version: str, min_firmware_version: str):
    def get_version_tuple(version_string: str):
        return tuple((int(part) for part in version_string.split(".")))

    if get_version_tuple(firmware_version) < get_version_tuple(min_firmware_version):
        raise RuntimeError(
            f"Firmware version {firmware_version} < {min_firmware_version}. Upgrade required."
        )


class LinWave:
    """
    Interface for linWave device.

    The device is controlled via TCP/IP:

    - Control port: 5432
    - Streaming ports: 5433 for channel 1 and 5434 for channel 2

    The interface is asynchronous and using `asyncio` for TCP/IP communication.
    This is especially beneficial for this kind of streaming applications,
    where most of the time the app is waiting for more data packets
    (`read more <https://realpython.com/async-io-python/>`_).
    Please refer to the examples for implementation details.
    """

    CHANNELS = (1, 2)  #: Available channels
    MAX_SAMPLERATE = 10_000_000  #: Maximum sampling rate in Hz
    PORT = 5432  #: Control port number
    RANGES = (0.05, 5.0)

    _DEFAULT_SETTINGS = _ChannelSettings(range_index=0, decimation=1)  #: Default settings
    _RANGE_INDEX: ClassVar[Dict[float, int]] = {
        0.05: 0,  # 50 mV
        5.0: 1,  # 5 V
    }  #: Mapping of range in volts and range index
    _MIN_FIRMWARE_VERSION = "2.2"

    def __init__(self, address: str):
        """
        Initialize device.

        Args:
            address: IP address of device.
                Use the method `discover` to get IP addresses of available linWave devices.

        Returns:
            Instance of `LinWave`

        Example:
            There are two ways constructing and using the `LinWave` class:

            1.  Without context manager, manually calling the `connect` and `close` method:

                >>> async def main():
                >>>     lw = waveline.LinWave("192.168.0.100")
                >>>     await lw.connect()
                >>>     print(await lw.get_info())
                >>>     ...
                >>>     await lw.close()
                >>> asyncio.run(main())

            2.  Using the async context manager:

                >>> async def main():
                >>>     async with waveline.LinWave("192.168.0.100") as lw:
                >>>         print(await lw.get_info())
                >>>         ...
                >>> asyncio.run(main())
        """
        self._address = address
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._connected = False
        self._recording = False
        self._channel_settings = {
            channel: replace(self._DEFAULT_SETTINGS) for channel in self.CHANNELS  # return copy
        }
        # wait for stream connections before start acq
        self._stream_connection_tasks: Set[asyncio.Task] = set()
        self._adc_to_volts = [1.5625e-06, 0.00015625]  # defaults, update after connect
        self._adc_to_eu = _adc_to_eu(self._adc_to_volts, self.MAX_SAMPLERATE)

    def __del__(self):
        if self._writer:
            self._writer.close()

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args, **kwargs):
        await self.close()

    @classmethod
    def discover(cls, timeout: float = 0.5) -> List[str]:
        """
        Discover linWave devices in network.

        Args:
            timeout: Timeout in seconds

        Returns:
            List of IP adresses
        """
        message = b"find"
        host = socket.gethostbyname(socket.gethostname())
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.bind((host, 0))  # random port for client
        sock.sendto(message, ("<broadcast>", cls.PORT))

        def get_response(timeout=timeout):
            sock.settimeout(timeout)
            try:
                while True:
                    _, (ip, _) = sock.recvfrom(1024)
                    yield ip
            except socket.timeout:
                return

        ip_addresses = list(get_response())
        if host in ip_addresses:
            ip_addresses.remove(host)

        return sorted(ip_addresses)

    @property
    def connected(self) -> bool:
        """Check if connected to device."""
        return self._connected

    async def connect(self):
        """Connect to device."""
        if self.connected:
            return

        logger.info(f"Open connection {self._address}:{self.PORT}...")
        self._reader, self._writer = await asyncio.open_connection(self._address, self.PORT)
        self._connected = True
        info = await self.get_info()
        logger.debug(f"Info: {info}")
        _check_firmware_version(info.firmware_version, self._MIN_FIRMWARE_VERSION)
        self._adc_to_volts = info.adc_to_volts
        self._adc_to_eu = _adc_to_eu(self._adc_to_volts, self.MAX_SAMPLERATE)

        logger.info("Set default settings...")
        await self.set_range(0, self.RANGES[self._DEFAULT_SETTINGS.range_index])
        await self.set_tr_decimation(0, self._DEFAULT_SETTINGS.decimation)

    async def close(self):
        """Close connection."""
        if not self.connected:
            return

        if self._recording:
            await self.stop_acquisition()

        logger.info(f"Close connection {self._address}:{self.PORT}...")
        try:
            self._writer.close()
            await self._writer.wait_closed()  # new in 3.7 -> might raise AttributeError
        except AttributeError:
            pass
        finally:
            self._connected = False

    @_require_connected
    async def _send_command(self, command):
        command_bytes = command.encode("utf-8") + b"\n"  # str -> bytes
        logger.debug("Send command: %a", command_bytes)
        self._writer.write(command_bytes)
        await self._writer.drain()

    @_require_connected
    async def _readline(self, timeout_seconds: Optional[float] = None) -> bytes:
        return await asyncio.wait_for(
            self._reader.readline(),  # type: ignore
            timeout=timeout_seconds,
        )

    @_require_connected
    async def _readlines(
        self,
        limit: Optional[int] = None,
        return_emptyline: bool = True,
    ) -> List[bytes]:
        lines: List[bytes] = []
        try:
            while True:
                # long timeout (1000 ms) for first line, then short timeouts (100 ms)
                timeout_seconds = 1.0 if not lines else 0.1
                line = await asyncio.wait_for(
                    self._reader.readline(),  # type: ignore
                    timeout=timeout_seconds,
                )
                lines.append(line)
                if limit and len(lines) >= limit:
                    break
                if return_emptyline and line == b"\n":
                    break
        except (TimeoutError, asyncio.TimeoutError):
            ...
        return lines

    @_require_connected
    async def identify(self, channel: int = 0):
        """
        Blink LEDs to identify device or single channel.

        Args:
            channel: Channel number (0 for all to identify device)

        Note:
            Available since firmware version 2.10.
        """
        self._check_channel_number(channel, allow_all=True)
        await self._send_command(f"identify @{channel:d}")

    @_require_connected
    async def get_info(self) -> Info:
        """Get device information."""
        await self._send_command("get_info")
        lines = await self._readlines()
        if not lines:
            raise RuntimeError("Could not get device information")

        info_dict = multiline_output_to_dict(lines)
        return Info(
            hardware_id=info_dict.get("hw_id", ""),
            firmware_version=info_dict["fw_version"],
            fpga_version=info_dict["fpga_version"],
            channel_count=as_int(info_dict["channel_count"], 0),
            input_range=(
                parse_array(info_dict["input_range"])
                if "input_range" in info_dict
                else ["50 mV", "5 V"]
            ),
            max_sample_rate=as_int(
                dict_get_first(
                    info_dict, ("max_sample_rate", "max_tr_sample_rate"), self.MAX_SAMPLERATE
                )
            ),
            adc_to_volts=[float(v) / 1e6 for v in parse_array(info_dict["adc2uv"])],
        )

    @_require_connected
    async def get_status(self) -> Status:
        """Get status information."""
        await self._send_command("get_status")
        lines = await self._readlines()
        if not lines:
            raise RuntimeError("Could not get status")

        status_dict = multiline_output_to_dict(lines)
        return Status(
            temperature=as_float(status_dict["temp"]),
            buffer_size=as_int(status_dict["buffer_size"]),
        )

    def _check_channel_number(self, channel: int, *, allow_all: bool = True):
        allowed_channels = (0, *self.CHANNELS) if allow_all else self.CHANNELS
        if channel not in allowed_channels:
            raise ValueError(
                f"Invalid channel number '{channel}'. "
                f"Select a channel from {allowed_channels} (0: all channels)"
            )

    @_require_connected
    async def get_setup(self, channel: int) -> Setup:
        """
        Get setup information.

        Args:
            channel: Channel number (0 for all channels)
        """
        self._check_channel_number(channel, allow_all=False)
        await self._send_command(f"get_setup @{channel:d}")
        lines = await self._readlines()
        if not lines:
            raise RuntimeError("Could not get setup")

        setup_dict = multiline_output_to_dict(lines)
        input_range = dict_get_first(setup_dict, ("input_range", "adc_range"))
        filter_setup = parse_filter_setup_line(setup_dict["filter"])
        return Setup(
            adc_range_volts=self.RANGES[as_int(input_range)],
            adc_to_volts=as_float(setup_dict["adc2uv"]) / 1e6,
            filter_highpass_hz=filter_setup[0],
            filter_lowpass_hz=filter_setup[1],
            filter_order=filter_setup[2],
            enabled=as_int(setup_dict["enabled"]) == 1,
            continuous_mode=as_int(setup_dict["cont"]) == 1,
            threshold_volts=as_float(setup_dict["thr"]) / 1e6,
            ddt_seconds=as_float(setup_dict["ddt"]) / 1e6,
            status_interval_seconds=as_float(setup_dict["status_interval"]) / 1e3,
            tr_enabled=as_int(setup_dict["tr_enabled"]) == 1,
            tr_decimation=as_int(setup_dict["tr_decimation"]),
            tr_pretrigger_samples=as_int(setup_dict["tr_pre_trig"]),
            tr_postduration_samples=as_int(setup_dict["tr_post_dur"]),
        )

    async def _set_range_index(self, channel: int, range_index: int):
        await self._send_command(f"set_adc_range {range_index:d} @{channel:d}")
        if channel > 0:
            self._channel_settings[channel].range_index = range_index
        else:
            self._channel_settings[1].range_index = range_index
            self._channel_settings[2].range_index = range_index

    @_require_connected
    async def set_range(self, channel: int, range_volts: float):
        """
        Set input range.

        Args:
            channel: Channel number (0 for all channels)
            range_volts: Input range in volts (0.05, 5)
        """
        self._check_channel_number(channel)
        try:
            range_index = self._RANGE_INDEX[range_volts]
        except KeyError:
            raise ValueError(f"Invalid range. Possible values: {self.RANGES}") from None

        logger.info(f"Set {_channel_str(channel)} range to {range_volts} V...")
        await self._set_range_index(channel, range_index)

    @_require_connected
    async def set_range_index(self, channel: int, range_index: int):
        """
        Set input range by index.

        Args:
            channel: Channel number (0 for all channels)
            range_index: Input range index (0: 0.05 V, 1: 5 V)
        """
        self._check_channel_number(channel)
        logger.info(f"Set {_channel_str(channel)} range to index {range_index}...")
        await self._set_range_index(channel, range_index)

    @_require_connected
    async def set_channel(self, channel: int, enabled: bool):
        """
        Enable/disable channel.

        Args:
            channel: Channel number (0 for all channels)
            enabled: Set to `True` to enable channel
        """
        self._check_channel_number(channel)
        await self._send_command(f"set_acq enabled {int(enabled)} @{channel:d}")

    @_require_connected
    async def set_continuous_mode(self, channel: int, enabled: bool):
        """
        Enable/disable continuous mode.

        Threshold will be ignored in continous mode.
        The length of the records is determined by `ddt` with `set_ddt`.

        Args:
            channel: Channel number (0 for all channels)
            enabled: Set to `True` to enable continuous mode
        """
        self._check_channel_number(channel)
        await self._send_command(f"set_acq cont {int(enabled)} @{channel:d}")

    @_require_connected
    async def set_ddt(self, channel: int, microseconds: int):
        """
        Set duration discrimination time (DDT).

        Args:
            channel: Channel number (0 for all channels)
            microseconds: DDT in µs
        """
        self._check_channel_number(channel)
        await self._send_command(f"set_acq ddt {int(microseconds)} @{channel:d}")

    @_require_connected
    async def set_status_interval(self, channel: int, seconds: int):
        """
        Set status interval.

        Args:
            channel: Channel number (0 for all channels)
            seconds: Status interval in s
        """
        self._check_channel_number(channel)
        await self._send_command(f"set_acq status_interval {int(seconds * 1e3)} @{channel:d}")

    @_require_connected
    async def set_tr_enabled(self, channel: int, enabled: bool):
        """
        Enable/disable recording of transient data.

        Args:
            channel: Channel number (0 for all channels)
            enabled: Set to `True` to enable transient data
        """
        self._check_channel_number(channel)
        await self._send_command(f"set_acq tr_enabled {int(enabled)} @{channel:d}")

    @_require_connected
    async def set_tr_decimation(self, channel: int, factor: int):
        """
        Set decimation factor of transient data and streaming data.

        The sampling rate will be 10 MHz / `factor`.

        Args:
            channel: Channel number (0 for all channels)
            factor: Decimation factor
        """
        self._check_channel_number(channel)
        factor = int(factor)
        await self._send_command(f"set_acq tr_decimation {factor:d} @{channel:d}")
        if channel > 0:
            self._channel_settings[channel].decimation = factor
        else:
            self._channel_settings[1].decimation = factor
            self._channel_settings[2].decimation = factor

    @_require_connected
    async def set_tr_pretrigger(self, channel: int, samples: int):
        """
        Set pre-trigger samples for transient data.

        Args:
            channel: Channel number (0 for all channels)
            samples: Pre-trigger samples
        """
        self._check_channel_number(channel)
        await self._send_command(f"set_acq tr_pre_trig {int(samples)} @{channel:d}")

    @_require_connected
    async def set_tr_postduration(self, channel: int, samples: int):
        """
        Set post-duration samples for transient data.

        Args:
            channel: Channel number (0 for all channels)
            samples: Post-duration samples
        """
        self._check_channel_number(channel)
        await self._send_command(f"set_acq tr_post_dur {int(samples)} @{channel:d}")

    @_require_connected
    async def set_filter(
        self,
        channel: int,
        highpass: Optional[float] = None,
        lowpass: Optional[float] = None,
        order: int = 8,
    ):
        """
        Set IIR filter frequencies and order.

        Args:
            channel: Channel number (0 for all channels)
            highpass: Highpass frequency in Hz (`None` to disable highpass filter)
            lowpass: Lowpass frequency in Hz (`None` to disable lowpass filter)
            order: Filter order
        """
        self._check_channel_number(channel)

        def khz_or_none(freq: Optional[float]):
            return freq / 1e3 if freq is not None else "none"

        await self._send_command(
            f"set_filter {khz_or_none(highpass)} {khz_or_none(lowpass)} {order} @{channel:d}"
        )

    @_require_connected
    async def set_threshold(self, channel: int, microvolts: float):
        """
        Set threshold for hit-based acquisition.

        Args:
            channel: Channel number (0 for all channels)
            microvolts: Threshold in µV
        """
        self._check_channel_number(channel)
        await self._send_command(f"set_acq thr {microvolts} @{channel:d}")

    @_require_connected
    async def start_acquisition(self):
        """Start data acquisition."""
        if self._recording:
            return

        if self._stream_connection_tasks:
            logger.debug("Wait for stream connections")
            await asyncio.wait(self._stream_connection_tasks)
            self._stream_connection_tasks.clear()

        logger.info("Start data acquisition...")
        await self._send_command("start_acq")
        self._recording = True

    @_require_connected
    async def stop_acquisition(self):
        """Stop data acquisition."""
        if not self._recording:
            return
        logger.info("Stop data acquisition...")
        await self._send_command("stop_acq")
        self._recording = False

    @_require_connected
    async def start_pulsing(
        self,
        channel: int,
        interval: float = 1,
        count: int = 4,
        cycles: int = 1,
    ):
        """
        Start pulsing.

        The number of pulses should be even, because pulses are generated by a square-wave signal
        (between LOW and HIGH) and the pulse signal should end LOW.

        Args:
            channel: Channel number (0 for all channels)
            interval: Interval between pulses in seconds
            count: Number of pulses per channel (should be even), 0 for infinite pulses
            cycles: Number of pulse cycles (automatically pulse through each channel in cycles).
                Only useful if all channels are chosen.
        """
        self._check_channel_number(channel)
        if count % 2 != 0:
            warn("Number of pulse counts should be even", stacklevel=1)
        logger.info(
            f"Start pulsing on {_channel_str(channel)} ("
            f"interval: {interval} s, "
            f"count: {count}, "
            f"cycles: {cycles})..."
        )
        await self._send_command(f"start_pulsing {interval} {count} {cycles} @{channel}")

    @_require_connected
    async def stop_pulsing(self):
        """
        Start pulsing.

        Args:

        """
        logger.info("Stop pulsing")
        await self._send_command("stop_pulsing")

    @_require_connected
    async def get_ae_data(self) -> List[AERecord]:
        """
        Get AE data records.

        Returns:
            List of AE data records (either status or hit data)
        """
        await self._send_command("get_ae_data")
        records = []
        while True:
            line = await self._readline(timeout_seconds=1)
            if line == b"\n":  # last line is an empty new line
                break

            logger.debug(f"Received AE data: {line}")

            record_type = line[:1]
            matches = dict(KV_PATTERN.findall(line))  # parse key-value pairs in line
            channel = int(matches[b"Ch"])
            range_index = self._channel_settings[channel].range_index
            adc_to_volts = self._adc_to_volts[range_index]
            adc_to_eu = self._adc_to_eu[range_index]

            if record_type in (b"H", b"S"):  # hit or status data
                record = AERecord(
                    type_=record_type.decode(),
                    channel=channel,
                    time=int(matches[b"T"]) / self.MAX_SAMPLERATE,
                    amplitude=int(matches.get(b"A", 0)) * adc_to_volts,
                    rise_time=int(matches.get(b"R", 0)) / self.MAX_SAMPLERATE,
                    duration=int(matches.get(b"D", 0)) / self.MAX_SAMPLERATE,
                    counts=int(matches.get(b"C", 0)),
                    energy=int(matches.get(b"E", 0)) * adc_to_eu,
                    trai=int(matches.get(b"TRAI", 0)),
                    flags=int(matches.get(b"flags", 0)),
                )
                records.append(record)
            elif record_type == b"R":  # marker record start
                ...
            else:
                logger.warning(f"Unknown AE data record: {line}")
        return records

    async def _read_tr_records(self, raw: bool) -> List[TRRecord]:
        records = []
        while True:
            headerline = await self._readline(timeout_seconds=1)
            if headerline == b"\n":  # last line is an empty new line
                break

            logger.debug(f"Received TR data: {headerline}")

            matches = dict(KV_PATTERN.findall(headerline))  # parse key-value pairs in line
            channel = int(matches[b"Ch"])
            samples = int(matches[b"NS"])
            range_index = self._channel_settings[channel].range_index
            adc_to_volts = self._adc_to_volts[range_index]

            data = np.frombuffer(
                await self._reader.readexactly(2 * samples),  # type: ignore
                dtype=np.int16,
            )
            assert len(data) == samples

            if not raw:
                data = np.multiply(data, adc_to_volts, dtype=np.float32)

            record = TRRecord(
                channel=channel,
                trai=int(matches.get(b"TRAI", 0)),
                time=int(matches.get(b"T", 0)) / self.MAX_SAMPLERATE,
                samples=samples,
                data=data,
                raw=raw,
            )
            records.append(record)
        return records

    @_require_connected
    async def get_tr_data(self, raw: bool = False) -> List[TRRecord]:
        """
        Get transient data records.

        Args:
            raw: Return TR amplitudes as ADC values if `True`, skip conversion to volts

        Returns:
            List of transient data records
        """
        await self._send_command("get_tr_data")
        return await self._read_tr_records(raw=raw)

    @_require_connected
    async def get_tr_snapshot(
        self, channel: int, samples: int, pretrigger_samples: int = 0, *, raw: bool = False
    ) -> List[TRRecord]:
        """
        Get snapshot of transient data.

        The recording starts with the execution of the command.
        The total number of samples is the sum of `samples` and the `pretrigger_samples`.
        The trai and time of the returned records are always `0`.

        Args:
            channel: Channel number (0 for all channels)
            samples: Number of samples to read
            pretrigger_samples: Number of samples to read before the execution of the command
            raw: Return TR amplitudes as ADC values if `True`, skip conversion to volts

        Returns:
            List of transient data records
        """
        self._check_channel_number(channel)
        decimations = {settings.decimation for settings in self._channel_settings.values()}
        if channel == 0 and len(decimations) > 1:
            raise ValueError(
                "TR decimation must be equal for all channels (current limitation of the firmware)"
            )
        assert len(decimations) == 1
        decimation = decimations.pop()
        samplerate = self.MAX_SAMPLERATE / decimation
        await asyncio.sleep(samples / samplerate)
        await self._send_command(f"get_tr_snapshot {(samples + pretrigger_samples):d} @{channel:d}")
        return await self._read_tr_records(raw=raw)

    async def acquire(
        self,
        raw: bool = False,
        poll_interval_seconds: float = 0.05,
    ) -> AsyncIterator[Union[AERecord, TRRecord]]:
        """
        High-level method to continuously acquire data.

        Args:
            raw: Return TR amplitudes as ADC values if `True`, skip conversion to volts
            poll_interval_seconds: Pause between data polls in seconds

        Yields:
            AE and TR data records

        Example:
            >>> async with waveline.LinWave("192.254.100.100") as lw:
            >>>     # apply settings
            >>>     await lw.set_channel(channel=1, enabled=True)
            >>>     await lw.set_channel(channel=2, enabled=False)
            >>>     await lw.set_range(channel=1, range_volts=0.05)
            >>>     async for record in lw.acquire():
            >>>         # do something with the data depending on the type
            >>>         if isinstance(record, waveline.AERecord):
            >>>             ...
            >>>         if isinstance(record, waveline.TRRecord):
            >>>             ...
        """
        await self.start_acquisition()
        try:
            while True:
                t = time.monotonic()
                for ae_record in await self.get_ae_data():
                    yield ae_record
                for tr_record in await self.get_tr_data(raw=raw):
                    yield tr_record
                t = time.monotonic() - t
                # allow other tasks to run, prevent blocking the event loop
                # https://docs.python.org/3/library/asyncio-task.html#sleeping
                await asyncio.sleep(max(0, poll_interval_seconds - t))
        finally:
            await self.stop_acquisition()

    @_require_connected
    def stream(
        self,
        channel: int,
        blocksize: int,
        *,
        raw: bool = False,
        timeout: Optional[float] = 5,
    ) -> AsyncIterator[Tuple[float, np.ndarray]]:
        """
        Async generator to stream channel data.

        Args:
            channel: Channel number [1, 2]
            blocksize: Number of samples per block
            raw: Return ADC values if `True`, skip conversion to volts
            timeout: Timeout in seconds

        Yields:
            Tuple of

            - relative time in seconds (first block: t = 0)
            - data as numpy array in volts (or ADC values if `raw` is `True`)

        Raises:
            TimeoutError: If TCP socket read exceeds `timeout`, usually because of buffer overflows

        Example:
            >>> async with waveline.LinWave("192.168.0.100") as lw:
            >>>     # apply settings
            >>>     await lw.set_range(0.05)
            >>>     await lw.set_filter(100e3, 500e3, 8)
            >>>     # open streaming port before start acq afterwards (order matters!)
            >>>     stream = lw.stream(channel=1, blocksize=65536)
            >>>     await lw.start_acquisition()
            >>>     async for time, block in stream:
            >>>         # do something with the data
            >>>         ...
        """
        self._check_channel_number(channel, allow_all=False)

        settings = self._channel_settings[channel]
        logger.info(
            (
                f"Start streaming acquisition on channel {channel} "
                f"(blocksize: {blocksize}, range: {self.RANGES[settings.range_index]} V)"
            )
        )

        port = int(self.PORT + channel)
        blocksize_bytes = int(blocksize * 2)  # 1 ADC value (16 bit) -> 2 * 8 byte
        to_volts = self._adc_to_volts[settings.range_index]
        interval = settings.decimation * blocksize / self.MAX_SAMPLERATE

        connection_task = asyncio.ensure_future(
            asyncio.open_connection(
                self._address,
                port,
                limit=blocksize_bytes,  # reduces CPU load by 10-20 % (default: 65536)
            ),
        )
        self._stream_connection_tasks.add(connection_task)

        class StreamGenerator:
            """Generator returning stream data with defined block size."""

            def __init__(self):
                self._time = 0
                self._connection_task = connection_task

            def __aiter__(self):
                return self

            async def get_reader_writer(self):
                if not self._connection_task.done():
                    await asyncio.wait([self._connection_task])
                return self._connection_task.result()

            async def aclose(self):
                _, writer = await self.get_reader_writer()
                writer.close()
                await writer.wait_closed()

            async def __anext__(self):
                reader, _ = await self.get_reader_writer()
                try:
                    buffer = await asyncio.wait_for(
                        reader.readexactly(blocksize_bytes),
                        timeout=timeout,
                    )
                    self._time += interval
                    data_adc = np.frombuffer(buffer, dtype=np.int16)
                    return (
                        self._time - interval,
                        data_adc if raw else np.multiply(data_adc, to_volts, dtype=np.float32),
                    )
                except asyncio.IncompleteReadError:
                    logger.info(f"Stop streaming on channel {channel}: EOF reached")
                    raise StopAsyncIteration from None

        return StreamGenerator()
