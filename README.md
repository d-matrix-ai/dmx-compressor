<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/d-matrix-ai/mltools/assets/139168891/e406e98a-51d7-48a4-a283-653be71900e6">
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/d-matrix-ai/mltools/assets/139168891/70f0aa39-139d-4f2e-932d-6c3fa1ee2926">
    <img alt="dmatrix-logo" src="https://github.com/d-matrix-ai/mltools/assets/139168891/e406e98a-51d7-48a4-a283-653be71900e6" width="900" height="180" style="max-width: 100%;"> 
  </picture>
</p>

<p align="center">
    <a href="https://github.com/d-matrix-ai/dmx-mltools/actions/workflows/python-app.yml">
        <img alt="Build" src="https://img.shields.io/github/actions/workflow/status/d-matrix-ai/dmx-mltools/python-app.yml">
    </a>
    <a href="https://github.com/d-matrix-ai/dmx-mltools/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/d-matrix-ai/dmx-mltools">
    </a>
    <a href="https://dmx-mltools.readthedocs.io/en/latest/">
        <img alt="Documentation" src="https://readthedocs.org/projects/dmx-mltools/badge/?version=latest">
    </a>
    <a href="https://github.com/d-matrix-ai/dmx-mltools/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/v/release/d-matrix-ai/dmx-mltools">
    </a>
    <a href="https://github.com/d-matrix-ai/dmx-mltools/commits/main">
        <img alt="Commits" src="https://img.shields.io/github/last-commit/d-matrix-ai/dmx-mltools/main">
    </a>
    <a href="https://github.com/d-matrix-ai/dmx-mltools/graphs/contributors">
        <img alt="Contributors" src="https://img.shields.io/github/contributors-anon/d-matrix-ai/dmx-mltools">
    </a>
    <a href="https://github.com/d-matrix-ai/dmx-mltools">
        <img alt="Stars" src="https://img.shields.io/github/stars/d-matrix-ai/dmx-mltools">
    </a>
</p>
<br/>

This project contains tools for deep neural net co-design for custom hardware accelerators.  

  - [Overview](#overview)
  - [Getting started](#getting-started)
  - [API in a nutshell](#api-in-a-nutshell)
  - [Next steps](#next-steps)

---

## Overview

In essence this is an extension of the [PyTorch](https://pytorch.org/) framework that implements hardware-efficient features, including 
- ***custom low-precision numerical formats and arithmetic***, 
- ***fine-grain structured weight sparsity***, and 
- ***custom operator approximation logic***.

In addition, the project provides a set of optimization tools for co-design using the above features.  


## Getting started

`pip install dmx-mltools`


## API in a nutshell

Given a PyTorch training/evaluation script, use the high-level API through two steps.

1. Monkey-patch PyTorch to be dmx-aware.  This is done by adding

    ```python
    from mltools import dmx
    dmx.aware()
    ```
    to the head of the script.  
    
    This augments `torch.nn` with the abovementioned features, while retaining all PyTorch functionalities, _i.e._

    - all valid PyTorch model definitions remain valid and
    - all PyTorch models remain functionally equivalent, in both forward and backward passes, to those in native PyTorch.

2. Wrap a DNN in a `dmx.Model` container, _e.g._
   
    ```python
    model = dmx.Model(Net())
    ```

    After this point, `model` is functionally equivalent to `Net()`, and all `torch` functionalities still available, but now `model` is ready for co-design configuration and/or optimization, at training time or post-training. 
    See advanced topics below for further details. 


## Next steps

For more detailed information, go over the following documents on specific topics.

- Configurations
- [Numerics](docs/numerics.rst)
- Weight sparsity
- Custom approximation logic

