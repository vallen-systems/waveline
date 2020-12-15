from datetime import datetime
from typing import List, Optional, Union

import numpy as np
from serial import Serial


class SpotWave:
    """
    Interface for spotWave devices.

    The USB-connected device exposes a virtual serial port for communication.
    """

    def __init__(self, port: Union[str, Serial]):
        ...

    @classmethod
    def discover(cls, timeout: float = 0.5) -> List[str]:
        """
        Discover connected spotWave devices.

        Args:
            timeout: Timeout in seconds

        Returns:
            List of port names
        """
        ...

    def get_status(self) -> str:
        ...

    def get_setup(self) -> str:
        ...

    def set_continuous_mode(self, enabled: bool):
        ...

    def set_ddt(self, microseconds: int):
        ...

    def set_status_interval(self, milliseconds: int):
        ...

    def set_tr_enabled(self, enabled: bool):
        ...

    def set_tr_decimation(self, factor: int):
        ...

    def set_tr_pretrigger(self, samples: int):
        ...

    def set_tr_postduration(self, samples: int):
        ...

    def set_tr_max_samples(self, samples: int):
        ...

    def set_cct(self, interval_seconds: int):
        ...

    def set_filter(self, highpass: float, lowpass: float, order: int = 4):
        ...

    def set_datetime(self, timestamp: Optional[datetime] = None):
        ...

    def set_threshold(self, decibel: float):
        ...

    def start_acquisition(self):
        ...
    
    def stop_acquisition(self):
        ...

    def get_ae_data(self):
        ...

    def get_tr_data(self):
        ...

    def get_data(self, samples: int) -> np.ndarray:
        ...
