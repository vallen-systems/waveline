import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--duration-acq",
        help="Acquisition duration for long-term acquisition tests",
        type=float,
        default=1,
    )
    parser.addoption(
        "--cwave-ip",
        help="IP address of specific conditionWave device",
        type=str,
    )


@pytest.fixture
def duration_acq(pytestconfig):
    return pytestconfig.getoption("duration_acq")


@pytest.fixture
def cwave_ip(pytestconfig):
    return pytestconfig.getoption("cwave_ip")
