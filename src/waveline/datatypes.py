"""
Common datatypes.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class AERecord:
    """AE data record, either status or hit data."""

    type_: str  #: Record type (hit or status data)
    channel: int  #: Channel number
    time: float  #: Time in seconds
    amplitude: float  #: Peak amplitude in volts
    rise_time: float  #: Rise time in seconds
    duration: float  #: Duration in seconds
    counts: int  #: Number of positive threshold crossings
    energy: float  #: Energy (EN 1330-9) in eu (1e-14 V²s)
    trai: int  #: Transient recorder index (key between `AERecord` and `TRRecord`)
    flags: int  #: Hit flags


@dataclass
class TRRecord:
    """Transient data record."""

    channel: int  #: Channel number
    trai: int  #: Transient recorder index (key between `AERecord` and `TRRecord`)
    time: float  #: Time in seconds
    samples: int  #: Number of samples
    data: np.ndarray  #: Array of transient data in volts (or ADC values if `raw` is `True`)
    raw: bool = False  #: ADC values instead of user values (volts)
