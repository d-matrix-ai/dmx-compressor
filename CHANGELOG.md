# Changelog

This file documents fixes/features added to dmx.compressor

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [v0.1.11]
  - torch.export FXIR tracing suppoer

## [v0.1.10]
  - torch >2.7 support
  - SLaNC advanced mode recipe, normalization layer extra parameters tuning
  - Updated GraphBuilder API for to_compiler_graph functions
  - other minor changes

## [v0.1.9]
  - SBFP weight storage option
  - updating dmir_compiler to dmx_compiler
  - VSIMD approximate functions in basic config
  - performance monitoring function
  - Input/output monitoring for dmxModules


## [v0.1.8]
 - Change pyproject.toml config to better support public release on PyPI.
 - Add support for experimental DMXModules
 - Improve tracing

## [v0.1.4]
- Support to export DMX modules as FXIR graphs

## [v0.1.2] (2023-11-08)

- Removed HuggingFace modules form DMX aware
- Added SmoothQuant quantization techniques
- Added Optimal Brain Compression activation calibration
- FX Transformation redesign
- Support for HuggingFace pipelines

## [v0.1.1] (2023-02-01)

- Internal version
