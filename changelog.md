# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Minimize and maximize buttons for device configuration dialog
- Documentation for sequencer instructions
- Function `plot_instruction` to plot values takes by a sequencer instruction
- Function `to_graph` to visualize sequencer instruction
- Sequencer instruction for ramping analog value

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
