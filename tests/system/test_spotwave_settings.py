from datetime import datetime

import pytest
from pytest import mark
from spotwave_fixture import sw

from waveline.utils import decibel_to_volts


def test_get_setup(sw):
    assert sw.get_setup()  # valid, non-empty response


def test_get_status(sw):
    assert sw.get_status()  # valid, non-empty response


@mark.parametrize("set_, expect", (
    (False, False),
    (True, True),
))
def test_set_continuous_mode(sw, set_, expect):
    sw.set_continuous_mode(set_)
    assert sw.get_setup().cont_enabled == expect


@mark.parametrize("set_, expect", (
    (-1, 0),
    (1000, 1000),
    (100_000, 100_000),
    (1_000_000, 1_000_000),  # TODO: should be set to max of 100_000
))
def test_set_ddt(sw, set_, expect):
    sw.set_ddt(set_)
    assert sw.get_setup().ddt_seconds == expect / 1e6


@mark.parametrize("set_, expect", (
    (-1, 0),
    (2000, 2000),
    (3600000, 3600000),
))
def test_set_status_interval(sw, set_, expect):
    sw.set_status_interval(set_)
    assert sw.get_setup().status_interval_seconds == expect / 1e3 


@mark.parametrize("set_, expect", (
    (False, False),
    (True, True),
))
def test_set_tr_enabled(sw, set_, expect):
    sw.set_tr_enabled(set_)
    assert sw.get_setup().tr_enabled == expect


@mark.parametrize("set_, expect", (
    (0, 0),
    (1, 1),
    (10, 10),
    (1_000_000, 1_000_000),
))
def test_set_tr_decimation(sw, set_, expect):
    sw.set_tr_decimation(set_)
    assert sw.get_setup().tr_decimation == expect


@mark.parametrize("set_, expect", (
    (0, 0),
    (2000, 2000),
    (10000, 10000),  # TODO: limit to 2000 / decimation
))
def test_set_tr_pretrigger(sw, set_, expect):
    sw.set_tr_pretrigger(set_)
    assert sw.get_setup().tr_pretrigger_samples == expect


@mark.parametrize("set_, expect, ddt", (
    (0, 0, 1000),
    (2000, 2000, 1000),  # TODO: limit to DDT
))
def test_set_tr_postduration(sw, set_, expect, ddt):
    sw.set_ddt(ddt)
    sw.set_tr_postduration(set_)
    assert sw.get_setup().tr_postduration_samples == expect


@mark.parametrize("set_, expect", (
    (0, 0),
    (-1, -1),
    (0.1, 0.1),
))
def test_set_cct(sw, set_, expect):
    sw.set_cct(set_)
    assert sw.get_setup().cct_seconds == expect


@mark.parametrize("set_, expect", (
    ((0, 0, 0), (0, 1000, 0)),
    ((0, 0, 4), (0, 1000, 0)),
    ((50, 300, 8), (50, 300, 8)),
    ((50, 1000, 8), (50, 1000, 8)),
    ((50, 2000, 8), (50, 1000, 8)),  # crop lowpass freq to nyquist
    ((50, 300, 3), (0, 1000, 0)),  # invalid order -> disable
    ((400, 300, 8), (0, 1000, 0)),  # invalid filter freqs -> disable
))
def test_set_filter(sw, set_, expect):
    sw.set_filter(highpass=set_[0], lowpass=set_[1], order=set_[2])
    setup = sw.get_setup()
    assert setup.filter_highpass_hz == expect[0] * 1e3
    assert setup.filter_lowpass_hz == expect[1] * 1e3
    assert setup.filter_order == expect[2]


def test_set_datetime(sw):
    timestamp = datetime(2020, 12, 17, 18, 12, 33)
    sw.set_datetime(timestamp)
    assert sw.get_status().datetime == timestamp


@mark.parametrize("set_, expect", (
    (-1, 0),
    (0, 0),
    (100, 100),
    (1_000_000, 1_000_000),
))
def test_threshold(sw, set_, expect):
    sw.set_threshold(set_)
    assert sw.get_setup().threshold_volts == set_ / 1e6
