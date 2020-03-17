import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

import coloredlogs
import numpy as np


MAX_SAMPLERATE = 10_000_000
CHANNELS = (1, 2)


logger_fmt = "%(asctime)s %(levelname)s %(message)s"
logger = logging.getLogger(__name__)

coloredlogs.install(
    level=logging.INFO,
    fmt=logger_fmt,
    datefmt="%H:%M:%S",
)


class ControlInterface:
    def __init__(self, address, port=5432):
        self.address = address
        self.port = port
        self.reader = None
        self.writer = None

    async def connect(self):
        logger.info(f"Open connection {self.address}:{self.port}...")
        self.reader, self.writer = await asyncio.open_connection(
            self.address, self.port,
        )

    async def _write(self, message):
        logger.debug("Write message: %s", message)
        self.writer.write(f"{message}\r".encode())
        await self.writer.drain()

    async def get_info(self):
        logger.info("Get info...")
        await self._write("get_info")
        data = await self.reader.read(1000)
        print(data.decode())

    async def set_decimation(self, factor: int):
        logger.info(f"Get decimation factor to {factor}...")
        for channel in CHANNELS:
            await self._write(f"set_decimation {channel:d} {factor:d}")

    async def start_acquisition(self):
        logger.info("Start data acquisition...")
        await self._write("start")
    
    async def stop_acquisition(self):
        logger.info("Stop data acquisition...")
        await self._write("stop")

    def __del__(self):
        logger.info("Close connection...")
        try:
            self.writer.close()
        except:
            pass


class StreamingInterface:
    def __init__(self, address):
        self.address = address

    async def read(self, blocksize: int, channel: int = 1):
        port = int(5432 + channel)
        reader, _ = await asyncio.open_connection(self.address, port)
        dt = np.int16
        # dt = dt.newbyteorder(">")
        while True:
            buffer = await reader.readexactly(blocksize * 2)  # 16 bit = 2 * 8 byte
            yield np.frombuffer(buffer, dtype=dt)


async def main(event_loop):
    parser = argparse.ArgumentParser(description="pyConditionWave")
    parser.add_argument("ip", help="IP address of conditionWave device")
    parser.add_argument("--samplerate", type=int, default=MAX_SAMPLERATE, help="Sample rate in Hz")
    parser.add_argument("--blocksize", type=int, default=1000000, help="Block size")

    args = parser.parse_args()

    control = ControlInterface(args.ip)
    await control.connect()
    await control.get_info()
    await control.set_decimation(int(MAX_SAMPLERATE / args.samplerate))
    await control.start_acquisition()

    stream = StreamingInterface(args.ip)
    fft = lambda y: np.abs(np.fft.rfft(y))

    with ThreadPoolExecutor(max_workers=1) as pool:
        async for y in stream.read(args.blocksize):
            Y = await loop.run_in_executor(pool, fft, y)
            y_max = np.max(y)
            cols = int(80 * y_max / 2**15)
            print(f"{y_max:<8d}" + "#" * cols + "-" * (80 - cols), end="\r")

    await control.stop_acquisition()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(loop))
    loop.close()
