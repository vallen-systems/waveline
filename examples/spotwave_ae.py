"""
Hit-based acquisition.

Acoustic Emission measurements are usually hit-based. Only when a defined threshold is crossed,
data is acquired. The acquired data is mainly influenced by two parameters:
- threshold amplitude
- duration discrimination time (DDT); if no threshold crossings are detected for the length of DDT,
  the end of the hit is determined

Additionally, status data can be acquired in defined intervals.

The following example shows a simple setup to acquire hits and status data with a spotWave device.
Hit data (AERecord) and transient data (TRRecord) are returned from different functions but can be
merged by matching the transient recorder index (trai) field in both records.
"""

import logging
from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np
from waveline import SpotWave
from waveline.spotwave import AERecord, TRRecord

logging.basicConfig(level=logging.INFO)


@dataclass
class HitRecord(AERecord):
    """All fields from AERecord + fields for transient data."""

    samples: int
    data: np.ndarray


def merge_ae_tr_records(generator):
    """Helper function to merge matching AERecords and TRRecords (same trai)."""
    dict_ae: Dict[int, AERecord] = {}
    dict_tr: Dict[int, TRRecord] = {}

    for record in generator:
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


def main():
    port = SpotWave.discover()[0]

    with SpotWave(port) as sw:
        print(sw.get_info())

        sw.set_continuous_mode(False)  # -> hit-based
        sw.set_ddt(10_000)  # 10.000 µs
        sw.set_status_interval(2)  # generate status data every 2 seconds
        sw.set_threshold(1000)  # 1000 µV = 60 dB(AE)
        sw.set_tr_enabled(True)  # enable transient data recording
        sw.set_tr_pretrigger(200)  # 200 pre-trigger samples
        sw.set_tr_postduration(0)  # 0 post-duration samples
        sw.set_filter(100e3, 450e3, 4)  # 100-450 kHz bandpass

        print(sw.get_setup())

        for record in merge_ae_tr_records(sw.acquire()):
            print(record)


if __name__ == "__main__":
    main()
