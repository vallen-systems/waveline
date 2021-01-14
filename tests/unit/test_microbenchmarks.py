from collections import defaultdict

import pytest


@pytest.fixture
def multiline_output():
    return [
        b"acq_enabled=1\n",
        b"log_enabled=0\n",
        b"adc2uv=1.74\n",
        b"cct=-0.5 s\n",
        b"filter=10.5-350 kHz, order 4\n",
        b"cont=0\n",
        b"thr=3162.5 uV\n",
        b"ddt=250  us\n",
        b"status_interval=1000 ms\n",
        b"tr_enabled=1\n",
        b"tr_decimation=2\n",
        b"tr_pre_trig=100\n",
        b"tr_post_dur=100\n",
        b"tr_max_samples=2097152\n",
    ]


def test_multiline_output_to_dict_orig(benchmark, multiline_output):
    def impl(lines):
        return defaultdict(
            str,
            [
                (lambda k, v="": [k.strip(), v.strip()])(*line.decode().split("=", maxsplit=1))
                for line in lines
            ],
        )

    benchmark(impl, multiline_output)


def test_multiline_output_to_dict_partition(benchmark, multiline_output):
    def impl(lines):
        def line_to_key_value(line: bytes):
            k, _, v = line.decode().partition("=")
            return k.strip(), v.strip()

        return defaultdict(str, [line_to_key_value(line) for line in lines])

    benchmark(impl, multiline_output)


def test_multiline_output_to_dict_split(benchmark, multiline_output):
    def impl(lines):
        def line_to_key_value(line: bytes):
            k, v, *_ = line.decode().split("=", maxsplit=1)
            return k.strip(), v.strip()

        return defaultdict(str, [line_to_key_value(line) for line in lines])

    benchmark(impl, multiline_output)
