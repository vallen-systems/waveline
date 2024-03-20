"""
Continuously compute and stream RMS values.

AE hit features (peak amplitude, rise time, duration, counts, energy) are directly computed on the
device with the maximum sampling rate of 2 MHz. RMS can be derived from energy and duration.
The computational load on the client side is minimal compared to transient data recording with
feature extraction on the raw data (2 MHz -> 4 MB/s).
"""

import logging
import math

from waveline import SpotWave

logging.basicConfig(level=logging.INFO)


def main():
    ports = SpotWave.discover()
    print(f"Discovered spotWave devices: {ports}")
    port = ports[0]

    with SpotWave(port) as sw:
        sw.set_datetime()
        sw.set_continuous_mode(True)  # enable continous mode
        sw.set_ddt(100_000)  # 100 ms block size
        sw.set_filter(highpass=50e3, lowpass=300e3, order=4)  # 50-300 kHz bandpass
        sw.set_status_interval(0)  # disable status data
        sw.set_tr_enabled(False)  # disable transient data

        for record in sw.acquire():
            # compute RMS from energy (in eu = 1e-14 V²s) and duration (in seconds)
            rms_volts = math.sqrt(record.energy / 1e14 / record.duration)
            print(f"RMS: {1e6 * rms_volts:0.2f} µV")


if __name__ == "__main__":
    main()
