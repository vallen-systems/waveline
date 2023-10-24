"""
Continuously stream transient data.

The continuous mode is activated with `set_continous_mode(True)`.
The data is acquired in chunks of equal length, defined by the the duration discrimination time.
The main differences between the continous mode and the streaming mode are following:
- in streaming mode, the raw data is pushed to TCP/IP sockets as raw binary data
- in continous mode, the data is pulled with following commands:
  - `get_ae_data`: AE features
  - `get_tr_data`: corresponding TR data / waveforms
"""

import argparse
import asyncio
import logging

from waveline import AERecord, LinWave, TRRecord

logging.basicConfig(level=logging.INFO)


async def main(ip: str):
    async with LinWave(ip) as lw:
        print(await lw.get_info())

        await lw.set_channel(channel=0, enabled=True)  # enabled all channels
        await lw.set_range_index(channel=0, range_index=0)  # set input range to 50 mV
        await lw.set_filter(channel=0, highpass=None, lowpass=500e3)  # 500 kHz lowpass
        await lw.set_continuous_mode(channel=0, enabled=True)  # enable continous mode
        await lw.set_ddt(channel=0, microseconds=100_000)  # set block size to 100 ms
        await lw.set_status_interval(channel=0, seconds=0)  # disable status data
        await lw.set_threshold(channel=0, microvolts=1_000)  # no effect for continous data
        await lw.set_tr_enabled(channel=0, enabled=True)  # enable transient data recording
        await lw.set_tr_decimation(channel=0, factor=5)  # decimation factor, 10 MHz / 5 = 2 MHz
        await lw.set_tr_pretrigger(channel=0, samples=0)  # no effect for continous data
        await lw.set_tr_postduration(channel=0, samples=0)  # no effect for continous data

        print(await lw.get_setup(channel=1))
        print(await lw.get_setup(channel=2))

        async for record in lw.acquire():
            if isinstance(record, TRRecord):
                print(record)
            elif isinstance(record, AERecord):
                ...  # ignore AE data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="linwave_ae")
    parser.add_argument("ip", help="IP address of linWave device")
    args = parser.parse_args()

    asyncio.run(main(args.ip))
