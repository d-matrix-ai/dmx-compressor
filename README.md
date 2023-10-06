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

The two methods getting started with mltools and mlreferences are either
installing via pip your python environment, or using git to checkout the source
and then installing the libraries in pip's developer mode.
    
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
model = corsair.Model(model)
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
- torch-mlir

### Examples

- Multi-layer perceptron on MNIST
- Transformer language models on GLUE and SQuAD
- Convolutional nets on CIFAR10/100 and Imagenet
