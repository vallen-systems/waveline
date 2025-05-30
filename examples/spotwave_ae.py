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

from __future__ import annotations

import logging

from waveline import SpotWave
from waveline.utils import HitMerger

logging.basicConfig(level=logging.INFO)


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

        with HitMerger(max_queue_size=None) as merger:
            for record in sw.acquire():
                hit = merger.process(record)
                if hit is not None:
                    print(hit)


if __name__ == "__main__":
    main()
