# ML tools

This project contains tools for deep neural net compression for Corsair deployment.

  - [Overview](#overview)
  - [Getting started](#getting-started)
  - [API in a nutshell](#api-in-a-nutshell)
  - [Next steps](#next-steps)

---

## Overview

In essence this is an extension of the [PyTorch](https://pytorch.org/) framework that implements Corsair-specific features, including 
- ***custom low-precision integer formats and arithmetic*** (for both activations and parameters), 
- ***fine-grain structured weight sparsity*** (for linear and convolutional weights), and 
- ***custom integer operation logics*** (for element-wise activation functions and normalizations).

It is so designed as 
- to provide a functionally complete set of Corsair-specific features to augment native PyTorch ones in order to make them Corsair-aware, and
- to retain all framework functionalities in order to support Corsair-aware network compression and optimization.  

Workloads compressed and optimized here are ***ML references***, to be converted into DMIR for downstream SW stack or functional simulator to consume.

### What is this project for?

- As a tool for algorithm research on Corsair-aware network compression and optimization
- As a tool for generation of ML references of customer workloads in production
- As an entry point for customer to optimize custom workloads for Corsair deployment

### What is this project ***not*** for?

- HW emulation
- Performance modeling and analysis

## Getting started

Follow these steps to set up as a d-MATRiX internal developer (not for external customers).  

1. Clone the repo.
   ```sh
   git clone git@git.d-matrix.ai:ml-team/mltools.git
   cd mltools
   ```
2. (*Recommended*) Start a new virtual environment with Python version 3.6 or higher, _e.g._
   ```sh
   conda create -n mltls python=3.9
   conda activate mltls
   ```
3. In the virtual environment set up the package and dependencies for development.

    ```sh
    pip install -e .
    ```
4. (*Recommended*) In the project root directory create a `.env` file for project-specific environmental variable settings such as common data and models directories.
    ```sh
    echo "DATA_PATH=/tools/d-matrix/ml/data" >> .env
    echo "MODEL_PATH=/tools/d-matrix/ml/models" >> .env
    ```

## API in a nutshell

Given a customer workload training/evaluation Python script, use the high-level API through two steps.

1. Import extended PyTorch that is Corsair-aware.  This is done by adding

    ```python
    corsair.aware()
    ```

    to the script.  This will augment `torch.nn` with Corsair-specific features, while retaining all PyTorch functionalities, _i.e._

    - all valid PyTorch model definitions remain valid and
    - all PyTorch models remain functionally equivalent, in both forward and backward passes, to those in native PyTorch.

2. Wrap a DNN in a `corsair.Model` container, _e.g._
   
    ```python
    model = corsair.Model(Net())
    ```

3. Define all Corsair-specific configurations in a `.yaml` file and transform a PyTorch `model` object by

    ```python
    model.transform("corsair_config.yaml")
    ```

    [This](configs/corsair_mnist_lenet.yaml) is an example Corsair configuration file.  

The following code blocks show a simplest example usage.  

<table>
<tr>
<th>Original evaluation script</th>
<th>Modified evaluation script</th>
</tr>
<tr>
<td>

```python
import torch​
​


data = data_loader()​
model = some_network()​

​​
results = evaluate(model, data)​
```

</td>
<td>

```python
import torch​
from mltools import corsair ​
​corsair.aware()

data = data_loader()​
model = some_network()​
model.transform('corsair_config.yaml') ​

results = evaluate(model, data)​
```

</td>
</tr>
</table>

## Next steps

For more detailed information, go over the following documents on specific topics.

### Corsair-aware features

- Corsair execution
- [Numerics](docs/numerics.rst)
- Weight sparsity
- Custom logic
- Configurations for Corsair-specific transformation
- List of supported modules and models
- DMIR

### Examples

- Multi-layer perceptron on MNIST
- Transformer language models on GLUE and SQuAD
- Convolutional nets on CIFAR10/100 and Imagenet
