# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- [conditionWave] all commands of new firmware (hit-based acquisition, pulsing, ...)
- [conditionWave] example for hit-based acquisition

### Changed

- [spotWave] Rename `stream` method to `acquire`. `stream` method is still an alias but deprecated and will be removed in the future

### Fixed

- [conditionWave] wait for all stream connection before `start_acquisition`

## [0.3.0] - 2021-06-15

### Added

- [conditionWave] multi-channel example
- [conditionWave] optional `start` argument (timestamp) for `stream`
- [spotWave] examples
- [spotWave] firmware check

### Changed

- [conditionWave] channel arguments for `set_range`, `set_decimation` and `set_filter`
- [spotWave] `set_status_interval` with seconds instead of milliseconds
- [spotWave] require firmware >= 00.25

### Removed
- [conditionWave] properties `input_range`, `decimation`, `filter_settings`

### Fixed

- [conditionWave] ADC to volts conversion factor
- [spotWave] aggregate TR/AE records to prevent IO timeouts

## [0.2.0] - 2020-12-18

First public release

[Unreleased]: https://github.com/vallen-systems/pyWaveLine/compare/0.3.0...HEAD
[0.3.0]: https://github.com/vallen-systems/pyWaveLine/compare/0.2.0...0.3.0
[0.2.0]: https://github.com/vallen-systems/pyWaveLine/releases/tag/0.2.0
