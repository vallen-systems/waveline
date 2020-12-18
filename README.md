# WaveLine

Library to easily interface with Vallen Systeme WaveLine™ devices using the public APIs:

- conditionWave
- spotWave

## Install

```
git clone https://github.com/vallen-systems/pyWaveline
cd pyWaveline
pip install -e .[dev]
```

Run tests (system tests excluded):

```
tox
```

The documentation is built with [sphinx](https://www.sphinx-doc.org):

```
cd docs
sphinx-build . _build
```

## Run system tests

System level tests are only available, if the targeted device can be discovered.

Run system tests with a spotWave device:
```
pytest tests/system/spotwave --duration-acq 1
```
