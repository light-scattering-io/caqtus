# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.16.0] - 2024-07-22

### Changed

- Camera configuration editor is now consistent with other device configuration editors

### Fixed

- Bug which caused device configurations to not always be saved in the device configuration dialog

## [4.15.0] - 2024-07-10

### Added

- Possibility to put complex transformations for device output

## [4.13.4] - 2024-07-09

### Fixed

- Crash when expressing sequence loops with incompatible units

## [4.13.0] - 2024-07-06

### Added

- Better event loop for SnapShot application

## [4.12.0] - 2024-07-06

### Added

- Better error handling and logging for experiment server and device server

## [4.11.0] - 2024-07-03

### Added

- Howto documentation for raising recoverable errors
- Documentation for camera device

### Changed

- Simplify camera device interface

## [4.10.3] - 2024-07-03

### Fixed

- Bug which crashed the application if the cell in a lane does not evaluate to the
  correct type

## [4.10.2] - 2024-07-03

### Fixed

- Don't display time lanes context menu when the sequence is not editable

## [4.10.1] - 2024-07-03

### Fixed

- Bug which caused Condetrol to crash when trying to interact with non-editable sequence

## [4.8.0] - 2024-06-30

### Added

- Documentation for recoverable errors

### Fixed

- Infinite recursion error when camera times out too many times

## [4.5.0] - 2024-06-30

### Added

- Highlight of sequence steps in error messages

## [4.4.0] - 2024-06-30

### Added

- Highlight of some caqtus objects in error messages

## [4.3.0] - 2024-06-30

### Added

- Better error message when an error occurs while connecting to a remote server

## [4.2.0] - 2024-06-30

### Added

- Better error messages when an exception occurs on a remote device

### Fixed

- Remote iteration on device server

## [4.1.0] - 2024-06-29

### Added

- Structured logging for device controller

### Fixed

- Used trio instead of asyncio for rpc server

## [4.0.0] - 2024-06-29

### Changed

- Display better error messages when an error occurs while running a sequence
- Replaced grpc remote devices with anyio remote devices

## [3.8.0] - 2024-06-28

### Added

- HOWTO for creating device extension
- Function to get a copy of caqtus serialization converter
- Export `FormDeviceConfigurationEditor`
  in `caqtus.gui.condetrol.device_configuration_editors`

### Changed

- Required methods of `Device` class are now abstract

## [3.7.0] - 2024-06-23

### Changed

- Shots are run in the order they are defined

## [3.6.0] - 2024-06-21

### Added

- Minimize and maximize buttons for device configuration dialog
- Documentation for sequencer instructions
- Function `plot_instruction` to plot values takes by a sequencer instruction
- Function `to_graph` to visualize sequencer instruction
- Sequencer instruction for ramping analog value

### Changed

- Load only sequence specific parameters by default

## [3.5.1] - 2024-06-16

### Fixed

- Bug which caused condetrol to crash sometimes when deleting a sequence step

## [3.5.0] - 2024-06-16

### Added

- Text representation of sequence steps when exporting to mime type `text/plain`

## [3.4.0] - 2024-06-14

### Added

- Drag/drop for multiple parameter steps at once
- Copy, paste, cut for sequence steps (Ctrl+C, Ctrl+V, Ctrl+X)
- Select all steps shortcut (Ctrl+A)
