import pytest
from pytest import mark

# all test coroutines will be treated as marked
pytestmark = pytest.mark.asyncio

async def test_connected(cw):
    assert cw.connected


async def test_get_info(cw):
    info = await cw.get_info()
    assert info.channel_count == 2
    assert info.range_count == 2
    assert info.max_sample_rate == 10_000_000


async def test_get_status(cw):
    await cw.get_status()


@mark.parametrize("channel", [1, 2])
class TestChannelSettings:
    async def test_get_setup(self, cw, channel):
        await cw.get_setup(channel)

    @mark.parametrize("range_volts", [0.05, 5])
    async def test_set_range(self, cw, channel, range_volts):
        await cw.set_range(channel, range_volts)
        assert (await cw.get_setup(channel)).adc_range_volts == range_volts

    @mark.parametrize("range_volts", [-1, 0, 0.5])
    async def test_set_range_invalid(self, cw, channel, range_volts):
        with pytest.raises(ValueError):
            await cw.set_range(channel, range_volts)

    @mark.parametrize(
        "set_, expect",
        (
            (-1, 1),
            (0, 1),
            (1, 1),
            (10, 10),
            (11.1, 11),
            (1_000_000, 1_000_000),  # TODO: limit to 500?
        ),
    )
    async def test_set_tr_decimation(self, cw, channel, set_, expect):
        await cw.set_tr_decimation(channel, set_)
        assert (await cw.get_setup(channel)).tr_decimation == expect

    @mark.parametrize(
        "set_, expect",
        (
            ((50, 300, 8), (50, 300, 8)),
            ((50, None, 8), (50, None, 8)),  # highpass
            ((50, 0, 8), (50, None, 8)),  # highpass
            ((None, 300, 8), (None, 300, 8)),  # lowpass
            ((0, 300, 8), (None, 300, 8)),  # lowpass
            ((None, None, 8), (None, None, 0)),  # bypass
            ((0, 0, 8), (None, None, 0)),  # bypass
            ((50, 10_000, 8), (50, None, 8)),  # lowpass freq > nyquist -> highpass
            # ((50, 300, 3), (None, None, 0)),  # invalid order -> disable
            ((400, 300, 8), (None, None, 0)),  # invalid filter freqs -> disable
        ),
    )
    async def test_set_filter(self, cw, channel, set_, expect):
        await cw.set_tr_decimation(channel, 1)  # max. sampling rate -> nyquist = 5 MHz
        await cw.set_filter(
            channel=channel,
            highpass=set_[0] * 1e3 if set_[0] else None,
            lowpass=set_[1] * 1e3 if set_[1] else None,
            order=set_[2],
        )
        setup = await cw.get_setup(channel)
        assert setup.filter_highpass_hz == (expect[0] * 1e3 if expect[0] else None)
        assert setup.filter_lowpass_hz == (expect[1] * 1e3 if expect[1] else None)
        assert setup.filter_order == expect[2]
