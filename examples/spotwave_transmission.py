"""
Measure transmission / time of flight with the synchronized CCT / pulser.

If both the sensor (spotWave sensor input) and the transmitter (spotWave CCT output) are mounted on
the same structure, the pulser functionality can be used to analyze transmission or measure the time
of flight (using thresholds or timepickers).
Setting a negativ value for CCT (or enabling the sync flag) will synchronize the pulser with the
first sample of the snapshot acquired with `get_data`.

The example requires matplotlib for plotting (install with `pip install matplotlib`).
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from waveline import SpotWave

logging.basicConfig(level=logging.INFO)


def main():
    port = SpotWave.discover()[0]

    # prepare plot
    plt.ion()
    _, ax = plt.subplots(figsize=(10, 3), tight_layout=True)

    with SpotWave(port) as sw:
        sw.set_cct(-1)  # sync pulse with get_tr_snapshot command
        sw.set_filter(90e3, 150e3, 4)  # 90-150 kHz bandpass

        while True:
            tr_record = sw.get_tr_snapshot(2048)  # read snapshot -> trigger pulser
            t = np.arange(tr_record.samples) / sw.CLOCK  # create time axis

            ax.clear()
            ax.plot(t * 1e6, tr_record.data * 1e6)
            ax.set_xlabel("Time [µs]")
            ax.set_ylabel("Amplitude [µV]")
            plt.pause(1)


if __name__ == "__main__":
    main()
