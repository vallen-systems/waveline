# WaveLine

[![CI](https://github.com/vallen-systems/pyWaveLine/workflows/CI/badge.svg)](https://github.com/vallen-systems/pyWaveLine/actions)
[![Documentation Status](https://readthedocs.org/projects/pywaveline/badge/?version=latest)](https://pywaveline.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/vallen-systems/pyWaveLine/badge.svg?branch=master)](https://coveralls.io/github/vallen-systems/pyWaveLine)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/waveline)](https://pypi.org/project/waveline)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/waveline)](https://pypi.org/project/waveline)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Python library to easily interface with Vallen Systeme WaveLine™ devices using the public APIs:

- conditionWave
- spotWave

## Documentation

Please visit http://pywaveline.rtfd.io for the documentation.

Check out the [examples](https://github.com/vallen-systems/pyWaveLine/tree/master/examples) for implementation details.

## Installation

Install the latest version from PyPI:

```
pip install waveline
```

Please note, that `waveline` requires Python 3.6 or newer. On Linux systems, `pip` is usually mapped to Python 2, so use `pip<version>` (e.g. `pip3` or `pip3.7`) instead. Alternatively, you can run `pip` from your specific Python version with `python<version> -m pip`.

## Contributing

Feature requests, bug reports and fixes are always welcome!

### Development setup

After cloning the repository, you can easily install the development environment and tools 
([pylint](https://www.pylint.org), [mypy](http://mypy-lang.org), [pytest](https://pytest.org), [tox](https://tox.readthedocs.io))
with. Using a [virtual environment](https://docs.python.org/3/library/venv.html) is strongly recommended.

```bash
git clone https://github.com/vallen-systems/pyWaveLine.git
cd pyWaveLine

# create virtual environment in directory .venv
python -m venv .venv
# activate virtual environment
source .venv/bin/activate  # linux
.venv\Scripts\activate  # windows

# install package (editable) and all development tools
pip install -e .[dev]
```

Run auto-formatter:

```
black .
isort .
```

Run linter:

```
pylint src/
```

Run the test suite in the current environment:

```
pytest
```

Run the CI pipeline (checks and tests) for all targeted (and installed) Python versions with tox:

```
tox
```

Build the documentation with [sphinx](https://www.sphinx-doc.org):

```bash
cd docs
sphinx-build . _build
```

### Run system tests

System level tests are only available, if the targeted device can be discovered.


Run system tests with a spotWave device:

```
pytest tests/system/spotwave --duration-acq 1
```

Measurement durations for long-term acquisition tests can be specified with the `--duration-acq` parameter (default: 1 second).

Run system tests with a conditionWave device (a specific IP can be provided with the `--cwave-ip` argument, otherwise the first discovered device will be used):

```
pytest tests/system/conditionwave --duration-acq 1 --cwave-ip 192.168.0.100
```
