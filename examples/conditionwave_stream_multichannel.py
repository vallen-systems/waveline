import asyncio
import logging
from datetime import datetime

from waveline import ConditionWave

logging.basicConfig(level=logging.INFO)


async def channel_acquisition(stream, channel: int):
    async for time, data in stream:
        print(f"Channel {channel}, {time}, {len(data)} samples")


async def main():
    ip = ConditionWave.discover()[0]

    async with ConditionWave(ip) as cw:
        await cw.set_decimation(channel=0, factor=10)  # 10 MHz / 10 = 1 MHz
        await cw.set_range(channel=0, range_volts=5)  # 5 V

        # create channel acquisition streams (wrapping async generators)
        streams = [
            channel_acquisition(
                cw.stream(channel, 1_000_000),
                channel,
            )
            for channel in (1, 2)
        ]

        try:
            await asyncio.gather(*streams, cw.start_acquisition())
        finally:
            await cw.stop_acquisition()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        ...
