"""
Hit-based acquisition.

Acoustic Emission measurements are usually hit-based. Only when a defined threshold is crossed,
data is acquired. The acquired data is mainly influenced by two parameters:
- threshold amplitude
- duration discrimination time (DDT); if no threshold crossings are detected for the length of DDT,
  the end of the hit is determined

Additionally, status data can be acquired in defined intervals.

The following example shows a simple setup to acquire hits and status data with a conditionWave
device. Hit data (AERecord) and transient data (TRRecord) are returned from different functions but
can be merged by matching the transient recorder index (trai) field in both records.
"""

import argparse
import asyncio
import logging
from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np

from waveline import ConditionWave, AERecord, TRRecord

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
    async with ConditionWave(ip) as cw:
        print(await cw.get_info())

        await cw.set_channel(channel=0, enabled=True)  # enabled all channels
        await cw.set_range(channel=0, range_volts=0.05)  # set input range to 50 mV
        await cw.set_continuous_mode(channel=0, enabled=False)  # -> hit-based
        await cw.set_status_interval(channel=0, seconds=2)  # generate status data every 2 seconds
        await cw.set_threshold(channel=0, microvolts=1_000)  # 1000 µV = 60 dB(AE)
        await cw.set_tr_enabled(channel=0, enabled=True)  # enable transient data recording
        await cw.set_tr_decimation(channel=0, factor=10)  # set decimation factor for transient data, 10 MHz / 10 = 1 MHz
        await cw.set_tr_pretrigger(channel=0, samples=200)  # 200 pre-trigger samples
        await cw.set_tr_postduration(channel=0, samples=200)  # 0 post-duration samples
        await cw.set_filter(channel=0, highpass=100e3, lowpass=450e3)  # 100-450 kHz bandpass

        print(await cw.get_setup(channel=1))
        print(await cw.get_setup(channel=2))

        async for record in merge_ae_tr_records(cw.acquire()):
            print(record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="conditionwave_ae")
    parser.add_argument("ip", help="IP address of conditionWave device")
    args = parser.parse_args()

    try:
        asyncio.run(main(args.ip))
    except KeyboardInterrupt:
        ...
