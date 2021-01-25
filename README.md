# Design Baselines for Model-Based Optimization

This repository contains several design baselines for model-based optimization. Our hope is that a common evaluation protocol will encourage future research and comparability in model-based design.

## Available Baselines

We provide the following list of baseline algorithms.

* Autofocused CbAS: `from design_baselines.autofocus import autofocus`
* Conditioning by Adaptive Sampling: `from design_baselines.cbas import cbas`
* Model Inversion Networks: `from design_baselines.mins import mins`
* Gradient Ascent: `from design_baselines.gradient_ascent import gradient_ascent`
* REINFORCE: `from design_baselines.reinforce import reinforce`
* Bo-qEI: `from design_baselines.bo_qei import bo_qei`
* CMA-ES: `from design_baselines.cma_es import cma_es`

## Setup

You can install the algorithms by cloning this repository and using anaconda.

```bash
REDACTED
conda env create -f design-baselines/environment.yml
```

## Usage

Every algorithm is implemented as a function that accepts a dictionary of hyper parameters called `config`. This makes interfacing with hyper parameter tuning platforms such as `ray.tune`, simple. For example, cbas can be called without using `ray.tune` with the following python code. 

```python
from design_baselines.cbas import cbas
cbas({
    "logging_dir": "data",
    "is_discrete": True,
    "normalize_ys": True,
    "normalize_xs": False,
    "task": "GFP-v0",
    "task_kwargs": {'seed': 0},
    "bootstraps": 5,
    "val_size": 200,
    "ensemble_batch_size": 100,
    "vae_batch_size": 100,
    "hidden_size": 256,
    "initial_max_std": 0.2,
    "initial_min_std": 0.1,
    "ensemble_lr": 0.001,
    "ensemble_epochs": 100,
    "latent_size": 32,
    "vae_lr": 0.001,
    "vae_beta": 5.0,
    "offline_epochs": 200,
    "online_batches": 10,
    "online_epochs": 10,
    "iterations": 50,
    "percentile": 80.0,
    "solver_samples": 128})
```

## Choosing Which Task

You may notice in the previous example that the `task` parameter is set to `HopperController-v0`. These baselines are tightly integrated with our [design-bench](REDACTED). These will automatically be installed with anaconda. For more information on which tasks are currently available for use, or how to register new tasks, please check out [design-bench](REDACTED).
# adaptive-mbo
