"""
Module for conditionWave device.

All device-related functions are exposed by the `ConditionWave` class.
"""

from warnings import warn

from .linwave import LinWave


class ConditionWave(LinWave):
    """
    Interface for conditionWave device.

    Deprecated:
        Please use `LinWave` class instead.
        The `ConditionWave` class will be removed in the future.
    """

    def __init__(self, *args, **kwargs):
        warn(
            "ConditionWave class has been renamed to LinWave and will be removed in the future",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
