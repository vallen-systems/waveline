import re
from typing import Dict, Iterable, List

# key = value pattern for ae/tr data
# fast(est) and simple, accept spaces around "="
# _KV_PATTERN = re.compile(br"(\w+)\s*=\s*(\S+)")
# accept words as keys w/o values; this seems next faster (incl. \S)?!
KV_PATTERN = re.compile(rb"([^\s=]+)(?:\s*=\s*(\S+))?")


def as_int(string, default: int = 0):
    """Return first sequence as int."""
    return int(string.strip().partition(" ")[0] or default)


def as_float(string, default: float = 0.0):
    """Return first sequence as float."""
    return float(string.strip().partition(" ")[0] or default)


def multiline_output_to_dict(lines: List[bytes]):
    """Helper function to parse output from get_info, get_status and get_setup."""
    return {k.strip(): v.strip() for k, _, v in [line.decode().partition("=") for line in lines]}


def dict_get_first(dict_: Dict, keys_desc_priority: Iterable[str], default=None, *, require=False):
    keys = tuple(keys_desc_priority)
    for key in keys:
        if key in dict_:
            return dict_[key]
    if require:
        raise KeyError(*keys)
    return default


def parse_array(line: str) -> List[str]:
    """Accept both space and comma as delimiters of values, prefer comma."""
    if "," in line:
        return [value.strip() for value in line.split(",")]
    return line.split()


def parse_filter_setup_line(line: str):
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

    def khz_or_none(k):
        try:
            return 1e3 * float(match.group(k))
        except:  # noqa
            return None

    return khz_or_none("hp"), khz_or_none("lp"), int(match.group("order"))
