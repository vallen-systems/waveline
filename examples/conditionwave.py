import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

import numpy as np

from waveline import ConditionWave


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


async def main(ip: str, samplerate: int, blocksize: int):
    async with ConditionWave(ip) as cw:
        print(await cw.get_info())
        await cw.set_range(0.05)
        await cw.set_decimation(int(cw.MAX_SAMPLERATE / samplerate))
        await cw.set_filter(100e3, 500e3, 8)
        await cw.start_acquisition()

        loop = asyncio.get_event_loop()

        with ThreadPoolExecutor(max_workers=1) as pool:
            async for timestamp, y in cw.stream(1, blocksize):
                # Y = await loop.run_in_executor(pool, lambda y: np.abs(np.fft.rfft(y)), y)
                y_max = np.max(y)
                cols = int(80 * y_max / cw.input_range)
                print(f"{y_max:<8f} V: " + "#" * cols + "-" * (80 - cols), end="\r")

        await cw.stop_acquisition()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pyConditionWave")
    parser.add_argument("ip", help="IP address of conditionWave device")
    parser.add_argument("--samplerate", "-s", type=int, default=ConditionWave.MAX_SAMPLERATE, help="Sample rate in Hz")
    parser.add_argument("--blocksize", "-b", type=int, default=1000000, help="Block size")

    args = parser.parse_args()

    asyncio.run(main(args.ip, args.samplerate, args.blocksize))
