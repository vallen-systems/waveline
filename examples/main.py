import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

import coloredlogs
import numpy as np

from conditionwave import ConditionWave, MAX_SAMPLERATE


logger_fmt = "%(asctime)s %(levelname)s %(message)s"
logger = logging.getLogger(__name__)

coloredlogs.install(
    level=logging.INFO,
    fmt=logger_fmt,
    datefmt="%H:%M:%S",
)


async def main(ip: str, samplerate: int, blocksize: int, event_loop):
    logger.info(f"Discovered devices: {ConditionWave.discover()}")

    cw = ConditionWave(ip)
    await cw.connect()
    await cw.get_info()
    await cw.set_range(0)
    await cw.set_decimation(int(MAX_SAMPLERATE / samplerate))
    await cw.start_acquisition()

    with ThreadPoolExecutor(max_workers=1) as pool:
        async for y in cw.stream(1, blocksize):
            # Y = await event_loop.run_in_executor(pool, lambda y: np.abs(np.fft.rfft(y)), y)
            y_max = np.max(y)
            cols = int(80 * y_max / cw.input_range)
            print(f"{y_max:<8f} V: " + "#" * cols + "-" * (80 - cols), end="\r")

    await cw.stop_acquisition()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pyConditionWave")
    parser.add_argument("ip", help="IP address of conditionWave device")
    parser.add_argument("--samplerate", "-s", type=int, default=MAX_SAMPLERATE, help="Sample rate in Hz")
    parser.add_argument("--blocksize", "-b", type=int, default=1000000, help="Block size")

    args = parser.parse_args()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(args.ip, args.samplerate, args.blocksize, loop))
    loop.close()
