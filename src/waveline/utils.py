"""Utility functions and classes."""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from waveline.datatypes import AERecord, TRRecord

logger = logging.getLogger(__name__)


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


@dataclass
class HitRecord:
    """Merged hit record combining AERecord and TRRecord."""

    ae: AERecord
    tr: TRRecord | None


class QueueFullError(Exception):
    """Exception raised when a queue is full."""


class HitMerger:
    """
    Merge AE and TR records into HitRecords based on the transient recorder index (trai).

    Since AE and TR records arrive in order for each channel, AE records are stored in
    channel-specific queues and merged with corresponding TR records as they become available.
    """

    @dataclass
    class ChannelState:
        queue: deque[AERecord]
        last_trai: int = 0

    def __init__(self, max_queue_size: int | None = None):
        """
        Initialize the HitMerger with an optional maximum queue size.

        Args:
            max_queue_size: Maximum queue size for each channel. If `None`, queues are unbounded.
        """
        self._channel_state: dict[int, HitMerger.ChannelState] = defaultdict(
            lambda: HitMerger.ChannelState(deque(maxlen=max_queue_size), 0)
        )

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.clear()

    def clear(self):
        """
        Clear all buffered AE records.
        """
        self._channel_state.clear()

    def process(self, record: AERecord | TRRecord) -> HitRecord | None:
        """
        Process a single AE or TR record.

        Returns:
            HitRecord if a merge occurred, otherwise None.

        Raises:
            QueueFullError: If the queue for a channel is full when processing an AERecord.
        """
        if isinstance(record, AERecord):
            return self._handle_ae_record(record)
        if isinstance(record, TRRecord):
            return self._handle_tr_record(record)
        return None

    def _handle_ae_record(self, ae_record: AERecord) -> HitRecord | None:
        if ae_record.trai == 0:
            return HitRecord(ae=ae_record, tr=None)

        state = self._channel_state[ae_record.channel]
        if state.queue.maxlen is not None and len(state.queue) >= state.queue.maxlen:
            raise QueueFullError()

        assert ae_record.trai > state.last_trai, "TRAI must be strictly increasing per channel"
        state.queue.append(ae_record)
        state.last_trai = ae_record.trai
        return None

    def _handle_tr_record(self, tr_record: TRRecord) -> HitRecord | None:
        state = self._channel_state[tr_record.channel]
        logger.debug("AE queue size for channel %d: %s", tr_record.channel, len(state.queue))

        while state.queue and state.queue[0].trai < tr_record.trai:
            ae_record = state.queue.popleft()
            logger.warning("Missing TR for TRAI %d, discard AE", ae_record.trai)

        if not state.queue or state.queue[0].trai > tr_record.trai:
            logger.warning("Missing AE for TRAI %d, discard TR", tr_record.trai)
            return None

        ae_record = state.queue.popleft()
        assert ae_record.trai == tr_record.trai
        assert ae_record.channel == tr_record.channel
        return HitRecord(ae=ae_record, tr=tr_record)
