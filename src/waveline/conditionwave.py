"""
Module for conditionWave device.

All device-related functions are exposed by the `ConditionWave` class.
"""

import asyncio
import logging
import socket
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from functools import wraps
from threading import Lock
from typing import AsyncIterator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class _ChannelSettings:
    """Channel settings."""

    range_volts: float  #: Input range in volts
    decimation: int  #: Decimation factor


class _AcquisitionStatus:
    """
    Helper class read and parse status data on control port during acquisition.

    Following messages are sent:
    - temp=<temperature in °C>
    - buffer_size=<buffer size in bytes>
    - error=<error message>
    """

    def __init__(self, stream_reader: asyncio.StreamReader):
        self._reader = stream_reader
        self._task = None
        self._lock = Lock()
        self._temperature = 0
        self._buffersize = 0

    async def _read_acquisition_status(self):
        logger.debug("Start reading acquisition status")
        try:
            while True:
                line = await self._reader.readuntil(b"\n")  # raises IncompleteReadError on EOF
                line = line.decode("utf-8").rstrip()

                try:
                    key, value = line.split("=")
                except ValueError:
                    logger.warning(f"Can not parse acqusition status '{line}'")

                if key == "temp":
                    # logger.debug(f"Temperature = {value} °C")
                    with self._lock:
                        self._temperature = int(value)
                elif key == "buffer_size":
                    # logger.debug(f"Buffer size = {value}")
                    with self._lock:
                        self._buffersize = int(value)
                elif key == "error":
                    logging.error(f"Error during acquisition: {value}")
                else:
                    raise logger.warning(f"Unknown status key '{key}'")
        except asyncio.IncompleteReadError:
            logger.warning("No more acquisition status to read, quit task")
        except asyncio.CancelledError:
            logger.debug("Stop reading acquisition status")

    async def start(self):
        """Start async task."""
        loop = asyncio.get_event_loop()  # workaround for Python 3.6
        self._task = loop.create_task(self._read_acquisition_status())

    async def stop(self):
        """Stop async task."""
        try:
            self._task.cancel()
            await self._task
        except asyncio.CancelledError:
            ...  # weird bug with Python 3.8 and Windows (Proactor event loop)
        finally:
            self._task = None

    def get_temperature(self):
        """Get system temperatur."""
        with self._lock:
            return self._temperature

    def get_buffersize(self):
        """Get current buffer size."""
        with self._lock:
            return self._buffersize


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
        self._reader = None
        self._writer = None
        self._connected = False
        self._daq_active = False
        self._daq_status: Optional[_AcquisitionStatus] = None
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
                    _, (ip, _) = sock.recvfrom(len(message))
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
        await self.set_decimation(0, self._DEFAULT_SETTINGS.decimation)

    async def close(self):
        """Close connection."""
        if not self.connected:
            return

        if self._daq_active:
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
    async def get_info(self) -> str:
        """Get device information."""
        logger.info("Get info...")
        await self._send_command("get_info")
        data = await self._reader.read(1000)  # type: ignore
        return data.decode()

    def _check_channel_number(self, channel: int):
        if channel not in (0, *self.CHANNELS):
            raise ValueError(
                f"Invalid channel number '{channel}'. "
                f"Select a single channel from {self.CHANNELS} or 0 for all"
            )

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
        await self._send_command(f"set_adc_range {channel:d} {range_index:d}")
        if channel > 0:
            self._channel_settings[channel].range_volts = range_volts
        else:
            self._channel_settings[1].range_volts = range_volts
            self._channel_settings[2].range_volts = range_volts

    @_require_connected
    async def set_decimation(self, channel: int, factor: int):
        """
        Set decimation factor.

        Args:
            channel: Channel number (0 for all channels)
            factor: Decimation factor [1, 500]
        """
        self._check_channel_number(channel)
        factor = int(factor)
        if not 1 <= factor <= 500:
            raise ValueError("Decimation factor must be in the range of [1, 500]")

        logger.info(f"Set {_channel_str(channel)} decimation factor to {factor}...")
        await self._send_command(f"set_decimation {channel:d} {factor:d}")
        if channel > 0:
            self._channel_settings[channel].decimation = factor
        else:
            self._channel_settings[1].decimation = factor
            self._channel_settings[2].decimation = factor

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

        def value_or(value: Optional[float], default_value: float):
            if value is None:
                return default_value
            return value

        if highpass is None and lowpass is None:
            logger.info(f"Set {_channel_str(channel)} filter to bypass")
            await self._send_command(f"set_filter {channel:d} 0")
        else:
            highpass_khz = value_or(highpass, 0) / 1e3  # 0 if None
            lowpass_khz = value_or(lowpass, 0.5 * self.MAX_SAMPLERATE) / 1e3  # nyquist if None

            logger.info(f"Set filter to {highpass_khz}-{lowpass_khz} kHz (order: {order})...")
            await self._send_command(f"set_filter {channel:d} {highpass_khz} {lowpass_khz} {order}")

    @_require_connected
    async def start_acquisition(self):
        """Start data acquisition."""
        if self._daq_active:
            return
        logger.info("Start data acquisition...")
        await self._send_command("start")
        self._daq_status = _AcquisitionStatus(self._reader)
        await self._daq_status.start()
        self._daq_active = True

    @_require_connected
    async def stream(
        self, channel: int, blocksize: int, *, start: Optional[datetime] = None
    ) -> AsyncIterator[Tuple[datetime, np.ndarray]]:
        """
        Async generator to stream channel data.

        Args:
            channel: Channel number [1, 2]
            blocksize: Number of samples per block
            start: Timestamp when acquisition was started with `start_acquisition`.
                Useful to get equal timestamps for multi-channel acquisition.
                If `None`, timestamp will be the time of the first acquired block.

        Yields:
            Tuple of datetime and numpy array (in volts)

        Example:
            >>> async with waveline.ConditionWave("192.168.0.100") as cw:
            >>>     # apply settings
            >>>     await cw.set_range(0.05)
            >>>     await cw.set_filter(100e3, 500e3, 8)
            >>>     # start daq and streaming
            >>>     await cw.start_acquisition()
            >>>     async for timestamp, block in cw.stream(channel=1, blocksize=65536):
            >>>         # do something with the data
            >>>         ...
        """
        if channel not in self.CHANNELS:
            raise ValueError(f"Channel must be in {self.CHANNELS}")
        if not self._daq_active:
            raise RuntimeError("Data acquisition not started")

        settings = self._channel_settings[channel]
        logger.info(
            (
                f"Start data acquisition on channel {channel} "
                f"(blocksize: {blocksize}, range: {settings.range_volts} V)"
            )
        )

        port = int(self.PORT + channel)
        blocksize_bytes = int(blocksize * 2)  # 1 ADC value (16 bit) -> 2 * 8 byte
        to_volts = settings.range_volts / 30_000  # 90 % of available 2^15 bytes

        timestamp = start
        interval = timedelta(seconds=settings.decimation * blocksize / self.MAX_SAMPLERATE)

        reader, writer = await asyncio.open_connection(self._address, port)
        while True:
            buffer = await reader.readexactly(blocksize_bytes)
            if timestamp is None:
                timestamp = datetime.now()
            data_adc = np.frombuffer(buffer, dtype=np.int16)
            data_volts = np.multiply(data_adc, to_volts, dtype=np.float32)
            yield timestamp, data_volts
            timestamp += interval
        writer.close()

    @_require_connected
    async def stop_acquisition(self):
        """Stop data acquisition."""
        if not self._daq_active:
            return
        logger.info("Stop data acquisition...")
        await self._send_command("stop")
        await self._daq_status.stop()
        self._daq_active = False

    @_require_connected
    def get_temperature(self) -> Optional[int]:
        """Get current device temperature in °C (only during acquisition)."""
        if not self._daq_active or self._daq_status is None:
            return None
        return self._daq_status.get_temperature()

    @_require_connected
    def get_buffersize(self) -> int:
        """Get buffer size in bytes (only during acquisition)."""
        if not self._daq_active or self._daq_status is None:
            return 0
        return self._daq_status.get_buffersize()

    def __del__(self):
        if self._writer:
            self._writer.close()
