# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.0] - 2024-03-20

### Added

- [spotwave] `SpotWave.get_tr_snapshot` method
- [spotwave] `SpotWave.start_pulsing`, `SpotWave.stop_pulsing` methods

### Changed

- Use common `Info`, `Status` and `Setup` dataclass for both spotWave and linWave
- [linwave] Deprecate `LinWave.set_range` method, use `LinWave.set_range_index` instead
- [linwave] Remove `Info.max_sample_rate` field
- [spotwave] Deprecate `SpotWave.get_data`, use `SpotWave.get_tr_snapshot` instead
- [spotwave] Remove `sync` parameter of `SpotWave.set_cct` method
- Remove deprecated `conditionwave` module

## [0.7.1] - 2023-10-18

### Fixed

- [spotwave] Return device names/paths from `SpotWave.discover` method, e.g. `/dev/ttyACM0` instead of `ttyACM0` on Linux systems

## [0.7.0] - 2023-10-17

### Added

- [linwave] `LinWave.set_range_index` method
- [linwave] `LinWave.identify` method to blind all LEDs or single channel to identify device/channel
- [linwave] `LinWave.get_tr_snapshot` method (experimental)
- [linwave] Add `hardware_id` to `get_info` output
- [spotwave] `LinWave.identify` method to blind LED
- Add Python 3.11 and 3.12 to CI pipeline
- Examples:
  - `linwave_pulsing`
  - `linwave_cont_tr`

### Changed

- [spotwave] Remove `cct_seconds` field in `get_setup` response `Setup`
- [spotwave] Make `readlines` method private
- List input range(s) in `get_info` response `Info.input_range` (human-readable format)
  - Remove `linwave.Info.range_count` field
  - Remove `spotwave.Info.input_range_decibel` field

## [0.6.0] - 2022-08-01

### Changed

- [linwave] Timeout parameter for `LinWave.stream` method (to detect buffer overflows), default: 5 s

### Fixed

- [linwave] Discovery port binding of client
- [linwave] Reduce CPU load in stream by setting TCP limit/buffer


## [0.5.0] - 2022-06-21

### Added

- `poll_interval_seconds` parameter for `SpotWave.acquire` and `ConditionWave.acquire` / `LinWave.acquire` method

### Changed

- Rename `ConditionWave` to `LinWave` (and the corresponding module `conditionwave` to `linwave`). The `ConditionWave` class is still an alias for the `LinWave` class but deprecated and will be removed in the future

### Fixed

- [linwave] Fix timeouts for multiline responses (`get_info`, `get_status`, `get_setup`)


## [0.4.1] - 2022-06-20

### Fixed

- [linwave] Increase TCP read timeout for `get_ae_data` / `get_tr_data` to prevent timeout errors


## [0.4.0] - 2022-05-17

### Added

- [linwave] Add all commands of new firmware (hit-based acquisition, pulsing, ...)
- [linwave] Add example for hit-based acquisition
- Add Python 3.9 and 3.10 to CI pipeline

### Changed

- [linwave] Rename `set_decimation` method to `set_tr_decimation`
- [linwave] Remove `get_temperature` and `get_buffersize` method (replace with `get_status` method)
- [spotWave] Rename `stream` method to `acquire`. `stream` method is still an alias but deprecated and will be removed in the future

### Fixed

- [linwave] Wait for all stream connection before `start_acquisition`


## [0.3.0] - 2021-06-15

### Added

- [linwave] Multi-channel example
- [linwave] Optional `start` argument (timestamp) for `stream`
- [spotWave] Add examples
- [spotWave] Add firmware check

### Changed

- [linwave] Channel arguments for `set_range`, `set_decimation` and `set_filter`
- [spotWave] `set_status_interval` with seconds instead of milliseconds
- [spotWave] Require firmware >= 00.25

### Removed
- [linwave] Properties `input_range`, `decimation`, `filter_settings`

### Fixed

- [linwave] ADC to volts conversion factor
- [spotWave] Aggregate TR/AE records to prevent IO timeouts


## [0.2.0] - 2020-12-18

First public release

[Unreleased]: https://github.com/vallen-systems/pyWaveLine/compare/0.8.0...HEAD
[0.8.0]: https://github.com/vallen-systems/pyWaveLine/compare/0.7.1...0.8.0
[0.7.1]: https://github.com/vallen-systems/pyWaveLine/compare/0.7.0...0.7.1
[0.7.0]: https://github.com/vallen-systems/pyWaveLine/compare/0.6.0...0.7.0
[0.6.0]: https://github.com/vallen-systems/pyWaveLine/compare/0.5.0...0.6.0
[0.5.0]: https://github.com/vallen-systems/pyWaveLine/compare/0.4.1...0.5.0
[0.4.1]: https://github.com/vallen-systems/pyWaveLine/compare/0.4.0...0.4.1
[0.4.0]: https://github.com/vallen-systems/pyWaveLine/compare/0.3.0...0.4.0
[0.3.0]: https://github.com/vallen-systems/pyWaveLine/compare/0.2.0...0.3.0
[0.2.0]: https://github.com/vallen-systems/pyWaveLine/releases/tag/0.2.0
