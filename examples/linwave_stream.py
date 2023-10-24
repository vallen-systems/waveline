import argparse
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from waveline import LinWave

logging.basicConfig(level=logging.INFO)


async def main(ip: str, samplerate: int, blocksize: int):
    async with LinWave(ip) as lw:
        print(await lw.get_info())
        await lw.set_range_index(channel=0, range_index=0)
        await lw.set_tr_decimation(channel=0, factor=int(lw.MAX_SAMPLERATE / samplerate))
        await lw.set_filter(channel=0, highpass=100e3, lowpass=500e3, order=8)

        stream = lw.stream(channel=1, blocksize=blocksize)  # open streaming port before start acq
        await lw.start_acquisition()

        with ThreadPoolExecutor(max_workers=1):
            asyncio.get_event_loop()
            async for _, y in stream:
                # execute (longer) blocking operations in the thread pool, e.g.:
                # Y = await loop.run_in_executor(pool, lambda y: np.abs(np.fft.rfft(y)), y)

                y_max = np.max(y)
                # visualize max amplitude with "level meter"
                cols = int(80 * y_max / 0.05)  # 50 mV input range
                print(f"{y_max:<8f} V: " + "#" * cols + "-" * (80 - cols), end="\r")

        await lw.stop_acquisition()


if __name__ == "__main__":
    print(f"Discovered devices: {LinWave.discover()}\n")

    parser = argparse.ArgumentParser(description="linwave_stream")
    parser.add_argument("ip", help="IP address of linWave device")
    parser.add_argument(
        "--samplerate",
        "-s",
        type=int,
        default=LinWave.MAX_SAMPLERATE,
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

    asyncio.run(main(args.ip, args.samplerate, args.blocksize))
