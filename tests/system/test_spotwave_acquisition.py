from time import sleep

import pytest
from pytest import mark
from spotwave_fixture import sw


@mark.parametrize("samples", (
    10,
    1024,
    65536,
    200_000,
))
def test_get_data(sw, samples):
    data = sw.get_data(samples)
    assert len(data) == samples


def test_acq_only_status(sw):
    status_interval_seconds = 0.01  # 10 ms

    sw.set_threshold(10_000_000)  # above range
    sw.set_continuous_mode(False)
    sw.set_tr_enabled(False)
    sw.set_status_interval(status_interval_seconds * 1e3)
    sw.clear_buffer()
    sw.start_acquisition()
    sleep(0.1)
    sw.stop_acquisition()

    ae_data = list(sw.get_ae_data())
    assert len(ae_data) == pytest.approx(9, abs=1)

    for i, record in enumerate(ae_data, start=1):
        assert record.time == i * status_interval_seconds
        assert record.type_ == "S"
        assert record.duration == status_interval_seconds
        assert record.trai == 0


@mark.parametrize("tr_enabled", (False, True))
def test_acq_continuous(sw, tr_enabled):
    ddt_seconds = 0.01  # 10 ms

    sw.set_continuous_mode(True)
    sw.set_ddt(ddt_seconds * 1e6)
    sw.set_tr_enabled(tr_enabled)
    sw.set_tr_decimation(1)
    sw.set_status_interval(0)
    sw.clear_buffer()
    sw.start_acquisition()
    sleep(0.1)
    sw.stop_acquisition()

    ae_data = list(sw.get_ae_data())
    assert len(ae_data) == pytest.approx(9, abs=1)

    for i, record in enumerate(ae_data, start=0):
        assert record.time == i * ddt_seconds
        assert record.type_ == "H"
        assert record.duration == ddt_seconds
        if tr_enabled:
            assert record.trai == i + 1
        else:
            assert record.trai == 0

    tr_data = list(sw.get_tr_data())
    if tr_enabled:
        assert len(tr_data) == pytest.approx(9, abs=1)
    else:
        assert len(tr_data) == 0
        return  # skip rest of test

    for i, record in enumerate(tr_data, start=0):
        assert record.trai == i + 1
        assert record.time == i * ddt_seconds
        assert record.samples == ddt_seconds * 2e6
