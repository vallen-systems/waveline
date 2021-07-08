import collections
from typing import List


def as_int(string, default: int = 0):
    """Return first sequence as int."""
    return int(string.strip().partition(" ")[0] or default)


def as_float(string, default: float = 0.0):
    """Return first sequence as float."""
    return float(string.strip().partition(" ")[0] or default)


def multiline_output_to_dict(lines: List[bytes]):
    """Helper function to parse output from get_info, get_status and get_setup."""
    return collections.defaultdict(
        str,
        [(k.strip(), v.strip()) for k, _, v in [line.decode().partition("=") for line in lines]],
    )
