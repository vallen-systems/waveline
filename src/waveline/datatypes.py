"""
Common datatypes.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class AERecord:
    """AE data record, either status or hit data."""

    #: Record type (`H` for hit or `S` for status data)
    type_: str
    #: Channel number
    channel: int
    #: Time in seconds (since `start_acq` command)
    time: float
    #: Peak amplitude in volts
    amplitude: float
    #: Rise time in seconds
    rise_time: float
    #: Duration in seconds
    duration: float
    #: Number of positive threshold crossings
    counts: int
    #: Energy (EN 1330-9) in eu (1e-14 V²s)
    energy: float
    #: Transient recorder index (key between `AERecord` and `TRRecord`)
    trai: int
    #: Hit flags
    flags: int


@dataclass
class TRRecord:
    """Transient data record."""

    #: Channel number
    channel: int
    #: Transient recorder index (key between `AERecord` and `TRRecord`)
    trai: int
    #: Time in seconds (since `start_acq` command)
    time: float
    #: Number of samples
    samples: int
    #: Array of transient data in volts (or ADC values if `raw` is `True`)
    data: np.ndarray
    #: ADC values instead of user values (volts)
    raw: bool = False


@dataclass
class Info:
    """Device information (static)."""

    #: Unique hardware id
    hardware_id: Optional[str]
    #: Firmware version
    firmware_version: str
    #: Number of channels
    channel_count: int
    #: List of selectable input ranges in human-readable format
    input_range: List[str]
    #: Conversion factors from ADC values to V for all input ranges
    adc_to_volts: List[float]
    #: Extra device information (specific to device and firmware version)
    extra: Dict[str, str]


@dataclass
class Status:
    """Status information."""

    #: Device temperature in °C
    temperature: float
    #: Flag if acquisition is active
    recording: bool
    #: Flag if pulsing is active
    pulsing: bool
    #: Extra status information (specific to device and firmware version)
    extra: Dict[str, str]


@dataclass
class Setup:
    """Channel setup."""

    #: Flag if channel is enabled
    enabled: bool
    #: Input range index of :attr:`Info.input_range` list
    input_range: int
    #: Conversion factor from ADC values to volts
    adc_to_volts: float
    #: Highpass frequency in Hz
    filter_highpass_hz: Optional[float]
    #: Lowpass frequency in Hz
    filter_lowpass_hz: Optional[float]
    #: Filter order
    filter_order: int
    #: Flag if continuous mode is enabled
    continuous_mode: bool
    #: Threshold for hit-based acquisition in volts
    threshold_volts: float
    #: Duration discrimination time (DDT) in seconds
    ddt_seconds: float
    #: Status interval in seconds
    status_interval_seconds: float
    #: Flag if transient data recording is enabled
    tr_enabled: bool
    #: Decimation factor for transient data
    tr_decimation: int
    #: Pre-trigger samples for transient data
    tr_pretrigger_samples: int
    #: Post-duration samples for transient data
    tr_postduration_samples: int
    #: Extra setup information (specific to device and firmware version)
    extra: Dict[str, str]
