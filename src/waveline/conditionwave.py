"""
Module for conditionWave device.

All device-related functions are exposed by the `ConditionWave` class.
"""

import asyncio
import logging
import socket
from dataclasses import dataclass, replace
from functools import wraps
from typing import AsyncIterator, List, Optional, Tuple

import numpy as np

from ._common import as_float, as_int, multiline_output_to_dict, parse_filter_setup_line

logger = logging.getLogger(__name__)

@dataclass
class Info:
    """Device information."""

    firmware_version: str  #: Firmware version
    fpga_version: str  #: FPGA version
    channel_count: int  #: Number of channels
    range_count: int  #: Number of selectable ranges
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

    range_volts: float  #: Input range in volts
    decimation: int  #: Decimation factor


def _require_connected(func):
    def check(obj: "ConditionWave"):
        if not obj.connected:
            raise ValueError("Device not connected")

    @wraps(func)
    async def async_wrapper(self: "ConditionWave", *args, **kwargs):
        check(self)
        return await func(self, *args, **kwargs)

    @wraps(func)
    def sync_wrapper(self: "ConditionWave", *args, **kwargs):
        check(self)
        return func(self, *args, **kwargs)

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


def _channel_str(channel: int) -> str:
    if channel == 0:
        return "all channels"
    return f"channel {channel:d}"


class ConditionWave:
    """
    Interface for conditionWave device.

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

    _DEFAULT_SETTINGS = _ChannelSettings(range_volts=0.05, decimation=1)  #: Default settings
    _RANGE_INDEX = {
        0.05: 0,  # 50 mV
        5.0: 1,  # 5 V
    }  #: Mapping of range in volts and range index

    def __init__(self, address: str):
        """
        Initialize device.

        Args:
            address: IP address of device.
                Use the method `discover` to get IP addresses of available conditionWave devices.

        Returns:
            Instance of `ConditionWave`

        Example:
            There are two ways constructing and using the `ConditionWave` class:

            1.  Without context manager, manually calling the `connect` and `close` method:

                >>> async def main():
                >>>     cw = waveline.ConditionWave("192.168.0.100")
                >>>     await cw.connect()
                >>>     print(await cw.get_info())
                >>>     ...
                >>>     await cw.close()
                >>> asyncio.run(main())

            2.  Using the async context manager:

                >>> async def main():
                >>>     async with waveline.ConditionWave("192.168.0.100") as cw:
                >>>         print(await cw.get_info())
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

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args, **kwargs):
        await self.close()

    @classmethod
    def discover(cls, timeout: float = 0.5) -> List[str]:
        """
        Discover conditionWave devices in network.

        Args:
            timeout: Timeout in seconds

        Returns:
            List of IP adresses
        """
        message = b"find"
        host = socket.gethostbyname(socket.gethostname())
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.bind((host, cls.PORT))
        sock.sendto(message, ("<broadcast>", cls.PORT))

        def get_response(timeout=timeout):
            sock.settimeout(timeout)
            while True:
                try:
                    _, (ip, _) = sock.recvfrom(1024)
                    yield ip
                except socket.timeout:
                    break

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

        logger.info("Set default settings...")
        await self.set_range(0, self._DEFAULT_SETTINGS.range_volts)
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
        timeout_seconds: Optional[float] = None,
    ) -> List[bytes]:
        lines = []
        while True:
            try:
                line = await asyncio.wait_for(
                    self._reader.readline(),  # type: ignore
                    timeout=timeout_seconds,
                )
                lines.append(line)
            except asyncio.TimeoutError:
                break
            if limit and len(lines) >= limit:
                break
        return lines

    async def get_info(self) -> Info:
        """Get device information."""
        await self._send_command("get_info")
        lines = await self._readlines(timeout_seconds=0.1)
        if not lines:
            raise RuntimeError("Could not get device information")

        info_dict = multiline_output_to_dict(lines)
        return Info(
            firmware_version=info_dict["fw_version"],
            fpga_version=info_dict["fpga_version"],
            channel_count=as_int(info_dict["channel_count"], 0),
            range_count=as_int(info_dict["range_count"], 0),
            max_sample_rate=as_int(info_dict["max_sample_rate"], 0),
            adc_to_volts=[float(v) / 1e6 for v in info_dict["adc2uv"].strip().split(" ")],
        )

    async def get_status(self) -> Status:
        """Get status information."""
        await self._send_command("get_status")
        lines = await self._readlines(timeout_seconds=0.1)
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

    async def get_setup(self, channel: int) -> Setup:
        """Get setup information."""
        self._check_channel_number(channel, allow_all=False)
        await self._send_command(f"get_setup @{channel:d}")
        lines = await self._readlines(timeout_seconds=0.1)
        if not lines:
            raise RuntimeError("Could not get setup")

        setup_dict = multiline_output_to_dict(lines)
        filter_setup = parse_filter_setup_line(setup_dict["filter"])
        return Setup(
            adc_range_volts=self.RANGES[as_int(setup_dict["adc_range"])],
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
        await self._send_command(f"set_adc_range {range_index:d} @{channel:d}")
        if channel > 0:
            self._channel_settings[channel].range_volts = range_volts
        else:
            self._channel_settings[1].range_volts = range_volts
            self._channel_settings[2].range_volts = range_volts

    async def set_tr_decimation(self, channel: int, factor: int):
        """
        Set decimation factor.

        Args:
            channel: Channel number (0 for all channels)
            factor: Decimation factor
        """
        self._check_channel_number(channel)
        factor = int(factor)
        # if not 1 <= factor <= 500:
        #     raise ValueError("Decimation factor must be in the range of [1, 500]")
        logger.info(f"Set {_channel_str(channel)} decimation factor to {factor}...")
        await self._send_command(f"set_acq tr_decimation {factor:d} @{channel:d}")
        if channel > 0:
            self._channel_settings[channel].decimation = factor
        else:
            self._channel_settings[1].decimation = factor
            self._channel_settings[2].decimation = factor

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

    async def start_acquisition(self):
        """Start data acquisition."""
        if self._recording:
            return
        logger.info("Start data acquisition...")
        await self._send_command("start_acq")
        self._recording = True

    @_require_connected
    async def stream(
        self, channel: int, blocksize: int, *, raw: bool = False
    ) -> AsyncIterator[Tuple[float, np.ndarray]]:
        """
        Async generator to stream channel data.

        Args:
            channel: Channel number [1, 2]
            blocksize: Number of samples per block
            raw: Return ADC values if `True`, skip conversion to volts

        Yields:
            Tuple of

            - relative time in seconds (first block: t = 0)
            - data as numpy array in volts (or ADC values if `raw` is `True`)

        Example:
            >>> async with waveline.ConditionWave("192.168.0.100") as cw:
            >>>     # apply settings
            >>>     await cw.set_range(0.05)
            >>>     await cw.set_filter(100e3, 500e3, 8)
            >>>     # start daq and streaming
            >>>     await cw.start_acquisition()
            >>>     async for time, block in cw.stream(channel=1, blocksize=65536):
            >>>         # do something with the data
            >>>         ...
        """
        self._check_channel_number(channel, allow_all=False)

        settings = self._channel_settings[channel]
        logger.info(
            (
                f"Start streaming acquisition on channel {channel} "
                f"(blocksize: {blocksize}, range: {settings.range_volts} V)"
            )
        )

        port = int(self.PORT + channel)
        blocksize_bytes = int(blocksize * 2)  # 1 ADC value (16 bit) -> 2 * 8 byte
        to_volts = settings.range_volts / 30_000  # 90 % of available 2^15 bytes

        time = 0.0
        interval = settings.decimation * blocksize / self.MAX_SAMPLERATE

        reader, writer = await asyncio.open_connection(self._address, port)
        while True:
            buffer = await reader.readexactly(blocksize_bytes)
            data_adc = np.frombuffer(buffer, dtype=np.int16)
            yield (
                time,
                data_adc if raw else np.multiply(data_adc, to_volts, dtype=np.float32),
            )
            time += interval
        writer.close()

    async def stop_acquisition(self):
        """Stop data acquisition."""
        if not self._recording:
            return
        logger.info("Stop data acquisition...")
        await self._send_command("stop_acq")

    def __del__(self):
        if self._writer:
            self._writer.close()
