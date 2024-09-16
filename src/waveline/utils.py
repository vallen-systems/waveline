"""Utility functions."""

from __future__ import annotations

import numpy as np


def decibel_to_volts(decibel: float | np.ndarray) -> float | np.ndarray:
    """
    Convert from dB(AE) to volts.

    Args:
        decibel: Input in decibel, scalar or array

    Returns:
        Input value(s) in volts
    """
    return 1e-6 * np.power(10, np.asarray(decibel) / 20)


def volts_to_decibel(volts: float | np.ndarray) -> float | np.ndarray:
    """
    Convert from volts to dB(AE).

    Args:
        volts: Inpult in volts, scalar or array

    Returns:
        Input value(s) in dB(AE)
    """
    return 20 * np.log10(np.asarray(volts) * 1e6)
