import asyncio
import logging
from datetime import datetime

from waveline import ConditionWave

logging.basicConfig(level=logging.INFO)


async def channel_acquisition(stream, channel: int):
    async for timestamp, data in stream:
        print(f"Channel {channel}, {timestamp}, {len(data)} samples")


async def main():
    ip = ConditionWave.discover()[0]

    async with ConditionWave(ip) as cw:
        await cw.set_decimation(channel=0, factor=10)  # 10 MHz / 10 = 1 MHz
        await cw.set_range(channel=0, range_volts=5)  # 5 V

        datetime_start = datetime.now()  # use start timestamp to synchronize timestamps of channels
        await cw.start_acquisition()

        # create channel acquisition streams (wrapping async generators)
        streams = [
            channel_acquisition(
                cw.stream(channel, 1_000_000, start=datetime_start),
                channel,
            )
            for channel in (1, 2)
        ]

        try:
            await asyncio.gather(*streams)
        finally:
            await cw.stop_acquisition()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        ...
