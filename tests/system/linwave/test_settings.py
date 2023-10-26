import pytest


async def test_connected(lw):
    assert lw.connected


async def test_get_info(lw):
    info = await lw.get_info()
    assert info.channel_count == 2
    assert info.input_range == ["50 mV", "5 V"]


async def test_get_status(lw):
    await lw.get_status()


@pytest.mark.parametrize("channel", [1, 2])
class TestChannelSettings:
    async def test_get_setup(self, lw, channel):
        await lw.get_setup(channel)

    @pytest.mark.parametrize("range_index", [0, 1])
    async def test_set_range_index(self, lw, channel, range_index):
        await lw.set_range_index(channel, range_index)
        assert (await lw.get_setup(channel)).input_range == range_index

    @pytest.mark.parametrize("range_index", [-1, 2])
    async def test_set_range_index_invalid(self, lw, channel, range_index):
        with pytest.raises(ValueError):
            await lw.set_range(channel, range_index)

    @pytest.mark.parametrize("enabled", [False, True])
    async def test_set_channel(self, lw, channel, enabled):
        await lw.set_channel(channel, enabled)
        assert (await lw.get_setup(channel)).enabled == enabled

    @pytest.mark.parametrize("enabled", [False, True])
    async def test_set_continuous_mode(self, lw, channel, enabled):
        await lw.set_continuous_mode(channel, enabled)
        assert (await lw.get_setup(channel)).continuous_mode == enabled

    @pytest.mark.parametrize("ddt_microseconds", [400, 10_000])
    async def test_set_ddt(self, lw, channel, ddt_microseconds):
        await lw.set_ddt(channel, ddt_microseconds)
        assert (await lw.get_setup(channel)).ddt_seconds == ddt_microseconds / 1e6

    @pytest.mark.parametrize(
        ("set_", "expect"),
        [
            (0, 0),
            (-1, 0),
            (1, 1),
            (100, 100),
        ],
    )
    async def test_set_status_interval(self, lw, channel, set_, expect):
        await lw.set_status_interval(channel, set_)
        assert (await lw.get_setup(channel)).status_interval_seconds == expect

    @pytest.mark.parametrize("enabled", [False, True])
    async def test_set_tr_enabled(self, lw, channel, enabled):
        await lw.set_tr_enabled(channel, enabled)
        assert (await lw.get_setup(channel)).tr_enabled == enabled

    @pytest.mark.parametrize(
        ("set_", "expect"),
        [
            (-1, 1),
            (0, 1),
            (1, 1),
            (10, 10),
            (11.1, 11),
        ],
    )
    async def test_set_tr_decimation(self, lw, channel, set_, expect):
        await lw.set_tr_decimation(channel, set_)
        assert (await lw.get_setup(channel)).tr_decimation == expect

    @pytest.mark.parametrize(
        ("set_", "expect"),
        [
            (0, 0),
            (-1, 0),
            (100, 100),
        ],
    )
    async def test_set_tr_pretrigger(self, lw, channel, set_, expect):
        await lw.set_tr_pretrigger(channel, set_)
        assert (await lw.get_setup(channel)).tr_pretrigger_samples == expect

    @pytest.mark.parametrize(
        ("set_", "expect"),
        [
            (0, 0),
            (-1, 0),
            (100, 100),
        ],
    )
    async def test_set_tr_postduration(self, lw, channel, set_, expect):
        await lw.set_tr_postduration(channel, set_)
        assert (await lw.get_setup(channel)).tr_postduration_samples == expect

    @pytest.mark.parametrize(
        ("set_", "expect"),
        [
            ((50, 300, 8), (50, 300, 8)),
            ((50, None, 8), (50, None, 8)),  # highpass
            ((50, 0, 8), (50, None, 8)),  # highpass
            ((None, 300, 8), (None, 300, 8)),  # lowpass
            ((0, 300, 8), (None, 300, 8)),  # lowpass
            ((None, None, 8), (None, None, 0)),  # bypass
            ((0, 0, 8), (None, None, 0)),  # bypass
            ((50, 10_000, 8), (50, None, 8)),  # lowpass > nyquist -> highpass
            ((400, 300, 8), (400, None, 8)),  # highpass > lowpass -> highpass
        ],
    )
    async def test_set_filter(self, lw, channel, set_, expect):
        await lw.set_tr_decimation(channel, 1)  # max. sampling rate -> nyquist = 5 MHz
        await lw.set_filter(
            channel=channel,
            highpass=set_[0] * 1e3 if set_[0] else None,
            lowpass=set_[1] * 1e3 if set_[1] else None,
            order=set_[2],
        )
        setup = await lw.get_setup(channel)
        assert setup.filter_highpass_hz == (expect[0] * 1e3 if expect[0] else None)
        assert setup.filter_lowpass_hz == (expect[1] * 1e3 if expect[1] else None)
        assert setup.filter_order == expect[2]
