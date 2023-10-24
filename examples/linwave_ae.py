"""
Hit-based acquisition.

Acoustic Emission measurements are usually hit-based. Only when a defined threshold is crossed,
data is acquired. The acquired data is mainly influenced by two parameters:
- threshold amplitude
- duration discrimination time (DDT); if no threshold crossings are detected for the length of DDT,
  the end of the hit is determined

Additionally, status data can be acquired in defined intervals.

The following example shows a simple setup to acquire hits and status data with a linWave
device. Hit data (AERecord) and transient data (TRRecord) are returned from different functions but
can be merged by matching the transient recorder index (trai) field in both records.
"""

import argparse
import asyncio
import logging
from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np
from waveline import AERecord, LinWave, TRRecord

logging.basicConfig(level=logging.INFO)


@dataclass
class HitRecord(AERecord):
    """All fields from AERecord + fields for transient data."""

    samples: int
    data: np.ndarray


async def merge_ae_tr_records(async_generator):
    """Helper function to merge matching AERecords and TRRecords (same trai)."""
    dict_ae: Dict[int, AERecord] = {}
    dict_tr: Dict[int, TRRecord] = {}

    async for record in async_generator:
        if isinstance(record, AERecord):
            if record.trai == 0:  # status data or hit without transient data -> return directly
                yield record
            else:
                dict_ae[record.trai] = record  # store in buffer to merge later
        if isinstance(record, TRRecord):
            dict_tr[record.trai] = record  # store in buffer to merge later

        # try to match and return merged HitRecords
        trais_ae = set(dict_ae.keys())
        trais_tr = set(dict_tr.keys())
        trais_match = trais_ae.intersection(trais_tr)
        for trai in trais_match:
            ae_record = dict_ae.pop(trai)
            tr_record = dict_tr.pop(trai)
            yield HitRecord(
                **asdict(ae_record),
                samples=tr_record.samples,
                data=tr_record.data,
            )


async def main(ip: str):
    async with LinWave(ip) as lw:
        print(await lw.get_info())

        await lw.set_channel(channel=0, enabled=True)  # enabled all channels
        await lw.set_range_index(channel=0, range_index=0)  # set input range to 50 mV
        await lw.set_filter(channel=0, highpass=100e3, lowpass=450e3)  # 100-450 kHz bandpass
        await lw.set_continuous_mode(channel=0, enabled=False)  # -> hit-based
        await lw.set_ddt(channel=0, microseconds=400)  # set duration discrimination time to 400 µs
        await lw.set_status_interval(channel=0, seconds=2)  # generate status data every 2 seconds
        await lw.set_threshold(channel=0, microvolts=1_000)  # 1000 µV = 60 dB(AE)
        await lw.set_tr_enabled(channel=0, enabled=True)  # enable transient data recording
        await lw.set_tr_decimation(channel=0, factor=10)  # decimation factor, 10 MHz / 10 = 1 MHz
        await lw.set_tr_pretrigger(channel=0, samples=200)  # 200 pre-trigger samples
        await lw.set_tr_postduration(channel=0, samples=200)  # 0 post-duration samples

        print(await lw.get_setup(channel=1))
        print(await lw.get_setup(channel=2))

        async for record in merge_ae_tr_records(lw.acquire()):
            print(record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="linwave_ae")
    parser.add_argument("ip", help="IP address of linWave device")
    args = parser.parse_args()

    asyncio.run(main(args.ip))
