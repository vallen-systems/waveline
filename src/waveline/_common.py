import logging
import re
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np

from waveline.datatypes import AERecord, Info, Setup, Status, TRRecord

logger = logging.getLogger(__name__)


def _check_firmware_version(firmware_version: str, min_firmware_version: str, base: int):
    def get_version_tuple(version_string: str):
        return tuple((int(part, base=base) for part in version_string.split(".")))

    if get_version_tuple(firmware_version) < get_version_tuple(min_firmware_version):
        raise RuntimeError(
            f"Firmware version {firmware_version} < {min_firmware_version}. Upgrade required."
        )


# key = value pattern for ae/tr data
# fast(est) and simple, accept spaces around "="
# _KV_PATTERN = re.compile(br"(\w+)\s*=\s*(\S+)")
# accept words as keys w/o values; this seems next faster (incl. \S)?!
_KV_PATTERN = re.compile(rb"([^\s=]+)(?:\s*=\s*(\S+))?")


def _adc_to_eu(adc_to_volts: float, samplerate: float) -> float:
    return adc_to_volts**2 * 1e14 / samplerate


def _parse_ae_headerline(
    line: bytes,
    samplerate: float,
    get_adc_to_volts_by_channel: Callable[[int], float],
    default_channel: Optional[int] = None,
) -> Optional[AERecord]:
    logger.debug("Parse AE data: %s", line)
    record_type = line[:1]
    matches = dict(_KV_PATTERN.findall(line))  # parse key-value pairs in line
    if record_type in (b"H", b"S"):  # hit or status data
        channel = int(matches[b"Ch"]) if default_channel is None else default_channel
        adc_to_volts = get_adc_to_volts_by_channel(channel)
        adc_to_eu = _adc_to_eu(adc_to_volts, samplerate)
        return AERecord(
            type_=record_type.decode(),
            channel=channel,
            time=int(matches[b"T"]) / samplerate,
            amplitude=int(matches.get(b"A", 0)) * adc_to_volts,
            rise_time=int(matches.get(b"R", 0)) / samplerate,
            duration=int(matches.get(b"D", 0)) / samplerate,
            counts=int(matches.get(b"C", 0)),
            energy=int(matches.get(b"E", 0)) * adc_to_eu,
            trai=int(matches.get(b"TRAI", 0)),
            flags=int(matches.get(b"flags", 0)),
        )
    if record_type == b"R":  # marker record start
        return None  # TODO
    logger.warning("Unknown AE data record: %s", line)
    return None


def _parse_tr_headerline(
    line: bytes,
    samplerate: float,
    default_channel: Optional[int] = None,
) -> TRRecord:
    logger.debug("Parse TR data: %s", line)
    matches = dict(_KV_PATTERN.findall(line))  # parse key-value pairs in line
    return TRRecord(
        channel=int(matches[b"Ch"]) if default_channel is None else default_channel,
        trai=int(matches.get(b"TRAI", 0)),
        time=int(matches.get(b"T", 0)) / samplerate,
        samples=int(matches[b"NS"]),
        data=np.empty(0, dtype=np.float32),
        raw=False,
    )


def _multiline_output_to_dict(lines: List[bytes]):
    """Helper function to parse output from get_info, get_status and get_setup."""
    return {
        key.strip(): value.strip()
        for key, sep, value in [line.decode().partition("=") for line in lines]
        if sep == "="
    }


def _dict_pop_first(
    dct: Dict, keys_desc_priority: Iterable[str], default: str = "", *, require=False
):
    keys = tuple(keys_desc_priority)
    for key in keys:
        if key in dct:
            return dct.pop(key)
    if require:
        raise KeyError(*keys)
    return default


def _strip_unit(s: str):
    """Return first sequence."""
    return s.strip().partition(" ")[0]


def _parse_array(line: str, allow_space: bool) -> List[str]:
    """Accept both comma and optionally space as delimiters of values, prefer comma."""
    if "," in line:
        return [value.strip() for value in line.split(",")]
    if allow_space:
        return line.split()
    if line:
        return [line]
    return []


def _is_number(s: str) -> bool:
    # https://stackoverflow.com/a/354130/9967707
    return s.replace(".", "", 1).isdigit()


def _parse_get_info_output(lines: List[bytes]) -> Info:
    def parse_input_range(s: str):
        return _parse_array(s, allow_space=False)

    def parse_adc_to_volts(s: str):
        return [float(v) / 1e6 for v in _parse_array(s, allow_space=True) if _is_number(v)]

    dct = _multiline_output_to_dict(lines)
    return Info(
        hardware_id=dct.pop("hw_id", None),
        firmware_version=dct.pop("fw_version"),
        channel_count=int(dct.pop("channel_count", "0"), 0),
        input_range=parse_input_range(dct.pop("input_range", "")),
        adc_to_volts=parse_adc_to_volts(dct.pop("adc2uv")),
        extra=dct,
    )


def _parse_get_status_output(lines: List[bytes]) -> Status:
    dct = _multiline_output_to_dict(lines)
    return Status(
        temperature=float(_strip_unit(dct.pop("temp", "0"))),
        recording=int(dct.pop("recording", "0")) == 1,
        pulsing=int(dct.pop("pulsing", "0")) == 1,
        extra=dct,
    )


def _parse_filter_setup_line(line: str):
    """
    Parse special filter setup row from get_setup.

    Example:
        10.5-350 kHz, order 4
        10.5-none kHz, order 4
        none-350 kHz, order 4
        none-none kHz, order 0
    """
    match = re.match(
        r"\s*(?P<hp>\S+)\s*-\s*(?P<lp>\S+)\s+.*o(rder)?\D*(?P<order>\d)",
        line,
        flags=re.IGNORECASE,
    )
    if not match:
        return None, None, 0

    def hz_or_none(k):
        try:
            return 1e3 * float(match.group(k))
        except:  # noqa
            return None

    return hz_or_none("hp"), hz_or_none("lp"), int(match.group("order"))


def _parse_get_setup_output(lines: List[bytes]) -> Setup:
    dct = _multiline_output_to_dict(lines)
    filter_setup = _parse_filter_setup_line(dct.pop("filter"))
    return Setup(
        enabled=int(dct.pop("enabled", "0")) == 1,
        input_range=int(_dict_pop_first(dct, ("input_range", "adc_range"), "0")),
        adc_to_volts=float(_strip_unit(dct.pop("adc2uv"))) / 1e6,
        filter_highpass_hz=filter_setup[0],
        filter_lowpass_hz=filter_setup[1],
        filter_order=filter_setup[2],
        continuous_mode=int(dct.pop("cont", "0")) == 1,
        threshold_volts=float(_strip_unit(dct.pop("thr", "0"))) / 1e6,
        ddt_seconds=float(_strip_unit(dct.pop("ddt", "0"))) / 1e6,
        status_interval_seconds=float(_strip_unit(dct.pop("status_interval", "0"))) / 1e3,
        tr_enabled=int(dct.pop("tr_enabled", "0")) == 1,
        tr_decimation=int(dct.pop("tr_decimation", "1")),
        tr_pretrigger_samples=int(dct.pop("tr_pre_trig", "0")),
        tr_postduration_samples=int(dct.pop("tr_post_dur", "0")),
        extra=dct,
    )
