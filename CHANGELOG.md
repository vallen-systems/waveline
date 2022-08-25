# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `LinWave.set_range_index` method


## [0.6.0] - 2022-08-01

### Changed

- Timeout parameter for `LinWave.stream` method (to detect buffer overflows), default: 5 s

### Fixed

- Discovery port binding of client
- Reduce CPU load in stream by setting TCP limit/buffer


## [0.5.0] - 2022-06-21

### Added

- `poll_interval_seconds` parameter for `SpotWave.acquire` and `ConditionWave.acquire` / `LinWave.acquire` method

### Changed

- Rename `ConditionWave` to `LinWave` (and the corresponding module `conditionwave` to `linwave`). The `ConditionWave` class is still an alias for the `LinWave` class but deprecated and will be removed in the future

### Fixed

- [conditionWave/linWave] Fix timeouts for multiline responses (`get_info`, `get_status`, `get_setup`)


## [0.4.1] - 2022-06-20

### Fixed

- [conditionWave] Increase TCP read timeout for `get_ae_data` / `get_tr_data` to prevent timeout errors


## [0.4.0] - 2022-05-17

### Added

- [conditionWave] Add all commands of new firmware (hit-based acquisition, pulsing, ...)
- [conditionWave] Add example for hit-based acquisition
- Add Python 3.9 and 3.10 to CI pipeline

### Changed

- [conditionWave] Rename `set_decimation` method to `set_tr_decimation`
- [conditionWave] Remove `get_temperature` and `get_buffersize` method (replace with `get_status` method)
- [spotWave] Rename `stream` method to `acquire`. `stream` method is still an alias but deprecated and will be removed in the future

### Fixed

- [conditionWave] Wait for all stream connection before `start_acquisition`


## [0.3.0] - 2021-06-15

### Added

- [conditionWave] Multi-channel example
- [conditionWave] Optional `start` argument (timestamp) for `stream`
- [spotWave] Add examples
- [spotWave] Add firmware check

### Changed

- [conditionWave] Channel arguments for `set_range`, `set_decimation` and `set_filter`
- [spotWave] `set_status_interval` with seconds instead of milliseconds
- [spotWave] Require firmware >= 00.25

### Removed
- [conditionWave] Properties `input_range`, `decimation`, `filter_settings`

### Fixed

- [conditionWave] ADC to volts conversion factor
- [spotWave] Aggregate TR/AE records to prevent IO timeouts


## [0.2.0] - 2020-12-18

First public release

[Unreleased]: https://github.com/vallen-systems/pyWaveLine/compare/0.6.0...HEAD
[0.6.0]: https://github.com/vallen-systems/pyWaveLine/compare/0.5.0...0.6.0
[0.5.0]: https://github.com/vallen-systems/pyWaveLine/compare/0.4.1...0.5.0
[0.4.1]: https://github.com/vallen-systems/pyWaveLine/compare/0.4.0...0.4.1
[0.4.0]: https://github.com/vallen-systems/pyWaveLine/compare/0.3.0...0.4.0
[0.3.0]: https://github.com/vallen-systems/pyWaveLine/compare/0.2.0...0.3.0
[0.2.0]: https://github.com/vallen-systems/pyWaveLine/releases/tag/0.2.0
