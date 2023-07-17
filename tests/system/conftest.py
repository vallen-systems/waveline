import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--duration-acq",
        help="Acquisition duration for long-term acquisition tests",
        type=float,
        default=1,
    )
    parser.addoption(
        "--linwave-ip",
        help="IP address of specific linWave device",
        type=str,
    )


@pytest.fixture()
def duration_acq(pytestconfig):
    return pytestconfig.getoption("duration_acq")


@pytest.fixture()
def linwave_ip(pytestconfig):
    return pytestconfig.getoption("linwave_ip")
