import argparse
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from waveline import ConditionWave

logging.basicConfig(level=logging.INFO)


async def main(ip: str, samplerate: int, blocksize: int):
    async with ConditionWave(ip) as cw:
        print(await cw.get_info())
        await cw.set_range(0.05)
        await cw.set_decimation(int(cw.MAX_SAMPLERATE / samplerate))
        await cw.set_filter(100e3, 500e3, 8)
        await cw.start_acquisition()

        with ThreadPoolExecutor(max_workers=1) as pool:
            loop = asyncio.get_event_loop()
            async for timestamp, y in cw.stream(1, blocksize):
                # execute (longer) blocking operations in the thread pool, e.g.:
                # Y = await loop.run_in_executor(pool, lambda y: np.abs(np.fft.rfft(y)), y)

                y_max = np.max(y)
                # visualize max amplitude with "level meter"
                cols = int(80 * y_max / 0.05)  # 50 mV input range
                print(f"{y_max:<8f} V: " + "#" * cols + "-" * (80 - cols), end="\r")

        await cw.stop_acquisition()


if __name__ == "__main__":
    print(f"Discovered devices: {ConditionWave.discover()}\n")

    parser = argparse.ArgumentParser(description="conditionwave_stream")
    parser.add_argument("ip", help="IP address of conditionWave device")
    parser.add_argument(
        "--samplerate",
        "-s",
        type=int,
        default=ConditionWave.MAX_SAMPLERATE,
        help="Sampling rate in Hz",
    )
    parser.add_argument(
        "--blocksize",
        "-b",
        type=int,
        default=1_000_000,
        help="Block size",
    )
    args = parser.parse_args()

    try:
        asyncio.run(main(args.ip, args.samplerate, args.blocksize))
    except (KeyboardInterrupt, SystemExit):
        logger.info("Graceful shutdown")
