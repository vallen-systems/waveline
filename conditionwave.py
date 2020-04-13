import asyncio
import logging
import warnings
import socket
from typing import List

import numpy as np


_CHANNELS = (1, 2)
_MAX_SAMPLERATE = 10_000_000
_RANGES = {
    0: 0.05,  # 50 mV
    1: 5.0,  # 5 V
}
_DEFAULT_RANGE = 0
_PORT = 5432


logger = logging.getLogger(__name__)


class ConditionWave:
    def __init__(self, address: str, port: int = _PORT):
        self._address = address
        self._port = port
        self._reader = None
        self._writer = None
        self._range = _RANGES[_DEFAULT_RANGE]
        self._decimation = 1
        self._daq_active = False

    @staticmethod
    def discover(timeout: float = 0.5) -> List[str]:
        MESSAGE = b"find"
        server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)

        # enable port reusage so we will be able to run multiple clients and servers on single (host, port) 
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        # enable broadcasting mode
        server.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        # bind to port
        server.bind(("", _PORT))

        # send broadcast message
        server.sendto(MESSAGE, ("<broadcast>", _PORT))

        def get_response(timeout=timeout):   
            server.settimeout(timeout)
            while True:
                try:
                    _, (ip, port) = server.recvfrom(len(MESSAGE))
                    yield ip
                except socket.timeout:
                    break

        return sorted(get_response())

    @property
    def input_range(self) -> float:
        return self._range

    @property
    def decimation(self) -> int:
        return self._decimation

    async def connect(self):
        logger.info(f"Open connection {self._address}:{self._port}...")
        self._reader, self._writer = await asyncio.open_connection(
            self._address, self._port,
        )

    async def _write(self, message):
        logger.debug("Write message: %s", message)
        self._writer.write(f"{message}\r".encode())
        await self._writer.drain()

    async def get_info(self):
        logger.info("Get info...")
        await self._write("get_info")
        data = await self._reader.read(1000)
        print(data.decode())

    async def set_range(self, range_index: int):
        if range_index not in _RANGES.keys():
            raise ValueError(f"Invalid range index.")
        logger.info(f"Set range to {range_index} ({_RANGES[range_index]} V)...")
        await self._write(f"set_adc_range 0 {range_index:d}")
        self._range = _RANGES[range_index]

    async def set_decimation(self, factor: int):
        factor = int(factor)
        logger.info(f"Set decimation factor to {factor}...")
        await self._write(f"set_decimation 0 {factor:d}")
        self._decimation = factor

    async def start_acquisition(self):
        logger.info("Start data acquisition...")
        await self._write("start")
        self._daq_active = True

    async def stream(self, channel: int, blocksize: int):
        if channel not in _CHANNELS:
            raise ValueError(f"Channel must be in {_CHANNELS}")
        if not self._daq_active:
            raise RuntimeError("Data acquisition not started")

        logger.info(
            (
                f"Start data acquisition on channel {channel} "
                f"(blocksize: {blocksize}, range: {self._range} V)"
            )
        )
        port = int(self._port + channel)
        blocksize_bits = int(blocksize * 2)  # 16 bit = 2 * 8 byte
        to_volts = float(self._range) / (2 ** 15)
        reader, _ = await asyncio.open_connection(self._address, port)
        while True:
            buffer = await reader.readexactly(blocksize_bits)
            yield np.frombuffer(buffer, dtype=np.int16).astype(np.float32) * to_volts
    
    async def stop_acquisition(self):
        logger.info("Stop data acquisition...")
        await self._write("stop")
        self._daq_active = False

    def __del__(self):
        logger.info("Close connection...")
        try:
            self._writer.close()
        except:
            pass
