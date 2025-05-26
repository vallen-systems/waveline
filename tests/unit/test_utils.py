import logging

import numpy as np
import pytest

from waveline import AERecord, TRRecord
from waveline.utils import HitMerger, QueueFullError, decibel_to_volts, volts_to_decibel


@pytest.mark.parametrize(
    ("input_", "output"),
    [
        (0, 1e-6),
        (100, 0.1),
        ([100, 120, 140], [0.1, 1, 10]),
    ],
)
def test_decibel_to_volts(input_, output):
    assert decibel_to_volts(input_) == pytest.approx(output)


@pytest.mark.parametrize(
    ("input_", "output"),
    [
        (1e-6, 0),
        (0.1, 100),
        ([0.1, 1, 10], [100, 120, 140]),
    ],
)
def test_volts_to_decibel(input_, output):
    assert volts_to_decibel(input_) == pytest.approx(output)


def make_ae(channel: int, trai: int):
    return AERecord(
        type="H",
        channel=channel,
        time=0,
        amplitude=0,
        rise_time=0,
        duration=0,
        counts=0,
        energy=0,
        trai=trai,
        flags=0,
    )


def make_tr(channel=1, trai=1):
    return TRRecord(
        channel=channel,
        trai=trai,
        time=0,
        samples=0,
        data=np.empty(0),
        raw=False,
    )


def test_hit_merger_max_queue_size():
    merger = HitMerger(max_queue_size=1)
    merger.process(make_ae(channel=1, trai=1))
    with pytest.raises(QueueFullError):
        merger.process(make_ae(channel=1, trai=2))


def test_hit_merger_no_trai():
    merger = HitMerger()
    ae = make_ae(channel=1, trai=0)
    hit = merger.process(ae)
    assert hit is not None
    assert hit.ae == ae
    assert hit.tr is None


def test_hit_merger_strictly_increasing_trai():
    merger = HitMerger()
    merger.process(make_ae(channel=1, trai=1))
    with pytest.raises(AssertionError):
        merger.process(make_ae(channel=1, trai=1))

    merger.process(make_ae(channel=2, trai=2))
    with pytest.raises(AssertionError):
        merger.process(make_ae(channel=2, trai=1))
    with pytest.raises(AssertionError):
        merger.process(make_ae(channel=2, trai=2))


def test_hit_merger_no_ae(caplog):
    merger = HitMerger()
    tr = make_tr(channel=1, trai=1)
    with caplog.at_level(logging.WARNING):
        hit = merger.process(tr)
        assert hit is None
    assert any("Missing AE for TRAI 1, discard TR" in msg for msg in caplog.messages)


def test_hit_merger_tr_before_ae(caplog):
    merger = HitMerger()
    tr = make_tr(channel=1, trai=1)
    ae = make_ae(channel=1, trai=2)

    with caplog.at_level(logging.WARNING):
        hit = merger.process(tr)
        assert hit is None
    assert any("Missing AE for TRAI 1, discard TR" in msg for msg in caplog.messages)

    assert merger.process(ae) is None


def test_hit_merger_missing_trai(caplog):
    merger = HitMerger()
    merger.process(make_ae(channel=1, trai=1))
    merger.process(make_ae(channel=1, trai=2))

    with caplog.at_level(logging.WARNING):
        assert merger.process(make_tr(channel=1, trai=2)) is not None
    assert any("Missing TR for TRAI 1, discard AE" in msg for msg in caplog.messages)


def test_hit_merger_ae_and_tr():
    merger = HitMerger()
    ae1 = make_ae(channel=1, trai=1)
    tr1 = make_tr(channel=1, trai=1)
    ae2 = make_ae(channel=2, trai=1)
    tr2 = make_tr(channel=2, trai=1)

    assert merger.process(ae1) is None
    assert merger.process(ae2) is None

    hit1 = merger.process(tr1)
    assert hit1 is not None
    assert hit1.ae == ae1
    assert hit1.tr is tr1

    hit2 = merger.process(tr2)
    assert hit2 is not None
    assert hit2.ae == ae2
    assert hit2.tr == tr2
