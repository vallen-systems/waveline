"""
Continuously stream transient data.

Streaming transient data with high sample rates introduces the risk of buffer overflows and loosing data.
The internal buffer has a size ~200.000 samples. One data record should not exceed 50 % of the buffer size.
About 25 % (~50.000 samples) is a good starting point for further optimizations.
With less samples, more hits are generated and the CPU load increases.

The record length is set by the duration discrimination time (DDT) with `set_ddt`.
"""

import logging
from dataclasses import asdict

import numpy as np
from waveline import SpotWave
from waveline.spotwave import AERecord, TRRecord

logging.basicConfig(level=logging.INFO)


def main():
    port = SpotWave.discover()[0]

    with SpotWave(port) as sw:
        # apply settings
        sw.set_continuous_mode(True)
        sw.set_status_interval(0)
        sw.set_tr_enabled(True)
        sw.set_tr_decimation(4)  # 2 MHz / 4 = 500 kHz
        sw.set_ddt(100_000)  # 100 ms * 500 kHz = 50.000 samples
        sw.set_filter(50e3, 200e3, 4)  # 50-200 kHz bandpass

        # show settings
        print("Settings:")
        for key, value in asdict(sw.get_setup()).items():
            print(f"- {key}: {value}")
        print()

        for record in sw.acquire():
            if isinstance(record, TRRecord):
                y = record.data
                y_max = np.max(y)

                # visualize max amplitude with "level meter"
                cols = int(80 * y_max / 0.05)  # 50 mV input range
                print(f"{y_max:<8f} V: " + "#" * cols + "-" * (80 - cols), end="\r")

            elif isinstance(record, AERecord):
                ...


if __name__ == "__main__":
    main()
