"""
Minimal pulsing example.

Both channels are acquired in streaming mode and a pulsing cycles with 4 pulses per channel is
started. This could be used as a coupling check for example.
"""


import argparse
import asyncio
import logging

import matplotlib.pyplot as plt
from waveline import LinWave

logging.basicConfig(level=logging.INFO)


async def main(ip: str):
    t = 0.1
    decimation = 5

    async with LinWave(ip) as lw:
        print(await lw.get_info())
        await lw.set_channel(channel=0, enabled=False)
        await lw.set_range_index(channel=0, range_index=0)
        await lw.set_filter(channel=0, highpass=None, lowpass=None, order=0)
        await lw.set_tr_decimation(channel=0, factor=decimation)

        stream_ch1 = lw.stream(channel=1, blocksize=int(t * LinWave.MAX_SAMPLERATE / decimation))
        stream_ch2 = lw.stream(channel=2, blocksize=int(t * LinWave.MAX_SAMPLERATE / decimation))
        await lw.start_acquisition()
        await lw.start_pulsing(channel=0, interval=0.01, count=4, cycles=1)

        _, y_ch1 = await anext(stream_ch1)
        _, y_ch2 = await anext(stream_ch2)

        await lw.stop_acquisition()

        _, ax = plt.subplots(figsize=(8, 4), tight_layout=True)
        ax.plot(y_ch1)
        ax.plot(y_ch2)
        ax.legend(("Ch1", "Ch2"))
        ax.set(xlabel="Samples", ylabel="Amplitude [V]")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="linwave_pulsing")
    parser.add_argument("ip", help="IP address of linWave device")
    args = parser.parse_args()

    asyncio.run(main(args.ip))
