<!-- <p align="center"> -->
<!--   <picture> -->
<!--     <source media="(prefers-color-scheme: dark)" srcset="https://github.com/d-matrix-ai/mltools/assets/139168891/e406e98a-51d7-48a4-a283-653be71900e6"> -->
<!--     <source media="(prefers-color-scheme: light)" srcset="https://github.com/d-matrix-ai/mltools/assets/139168891/70f0aa39-139d-4f2e-932d-6c3fa1ee2926"> -->
<!--     <img alt="dmatrix-logo" src="https://github.com/d-matrix-ai/mltools/assets/139168891/e406e98a-51d7-48a4-a283-653be71900e6" width="900" height="180" style="max-width: 100%;">  -->
<!--   </picture> -->
<!-- </p> -->

<p align="center">
    <a href="https://github.com/d-matrix-ai/dmx-compressor/actions/workflows/python-app.yml">
        <img alt="Build" src="https://img.shields.io/github/actions/workflow/status/d-matrix-ai/dmx-compressor/python-app.yml">
    </a>
    <a href="https://github.com/d-matrix-ai/dmx-compressor/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/d-matrix-ai/dmx-compressor">
    </a>
    <a href="https://dmx-compressor.readthedocs.io/en/latest/">
        <img alt="Documentation" src="https://readthedocs.org/projects/dmx-compressor/badge/?version=latest">
    </a>
    <a href="https://github.com/d-matrix-ai/dmx-compressor/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/v/release/d-matrix-ai/dmx-compressor">
    </a>
    <a href="https://github.com/d-matrix-ai/dmx-compressor/commits/main">
        <img alt="Commits" src="https://img.shields.io/github/last-commit/d-matrix-ai/dmx-compressor/main">
    </a>
    <a href="https://github.com/d-matrix-ai/dmx-compressor/graphs/contributors">
        <img alt="Contributors" src="https://img.shields.io/github/contributors-anon/d-matrix-ai/dmx-compressor">
    </a>
    <a href="https://github.com/d-matrix-ai/dmx-compressor">
        <img alt="Stars" src="https://img.shields.io/github/stars/d-matrix-ai/dmx-compressor">
    </a>
</p>
<br/>

This project contains tools for deep neural net co-design for custom hardware accelerators.  

  - [Overview](#overview)
  - [Getting started](#getting-started)
  - [Usage](#usage)
    - [Basic API](basic-api)
    - [Hugging Face pipeline API](hugging-face-pipeline-api)
  - [Next steps](#next-steps)

---

## Overview

In essence this is an extension of the [PyTorch](https://pytorch.org/) framework that implements hardware-efficient features, including 
- ***custom low-precision numerical formats and arithmetic***, 
- ***fine-grain structured weight sparsity***, and 
- ***custom operator approximation logic***.

In addition, the project provides a set of optimization tools for co-design using the above features.  


## Getting started

`pip install -compressor`


## Usage

### Basic API

Given a PyTorch model, _e.g._ `Net()`, wrap it in a `dmx.Model` container: 

```python
from dmx import compressor

model = compressor.Model(Net())
```

Here `model` is functionally equivalent to `Net()`, and all `torch` functionalities are still available, but `model` is equipped with d-Matrix specific features, making it ready for co-design configuration and/or optimization, at training time or post-training. 
See advanced topics for further details. 

`model.dmx_config` is a dictionary that contains all, and only those, configurations that affect the functional behavior of the model, different from the behavior of the original `Net()`. 
Use method `model.transform()` to set these configurations, through application of configuration rules. 
See advanced topics for engineering of configuration rules.  

There are two predefined special rule sets `compressor.config_rules.BASELINE` and `compressor.config_rules.BASIC`; the former is a dummy that does not change the original model's functional behavior, whereas the latter brings the model to a functional state that is equivalent to basic-mode execution on d-Matrix's hardware, _e.g._ 

```python
model = model.transform(
    model.dmx_config,
    *compressor.config_rules.BASIC,
)
```

### Hugging Face pipeline API

To leverage the popularity of [Hugging Face's pipeline API for inference](https://huggingface.co/docs/transformers/en/pipeline_tutorial), we extend `transformers.pipeline()` to `compressor.pipeline()`, which retains all existing functionality of pipelines while enabling model transformation and configuration for deployment on d-Matrix hardware.  

```python
from dmx.compressor import pipeline

pipe = pipeline(
    task="text-generation",
    model="openai-community/gpt2-xl",
    dmx_config="BASIC",  # make the model deployable on d-Matrix backend
    ...
)

# Deploy pipe the same way as Hugging Face provides.
```


## Next steps

For more detailed information, go over the following documents on specific topics.

- Configurations
- [Numerics](docs/numerics.rst)
- Weight sparsity
- Custom approximation logic

