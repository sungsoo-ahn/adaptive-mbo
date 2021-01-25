from ray import tune
import click
import ray
import os
from random import randint


@click.group()
def cli():
    """A group of experiments for training Conservative Score Models
    and reproducing our ICLR 2021 results.
    """


#############


@cli.command()
def test():
    from design_baselines.diffusion import diffusion
    config = {
            "logging_dir": "data",
            "task": "HopperController-v0",
            "task_kwargs": {},
            "is_discrete": False,
            "normalize_ys": True,
            "normalize_xs": True,
            "continuous_noise_std": 0.0,
            "val_size": 200,
            "batch_size": 128,
            "epochs": 500,
            "activations": ["leaky_relu", "leaky_relu"],
            "hidden_size": 2048,
            "initial_max_std": 0.2,
            "initial_min_std": 0.1,
            "forward_model_lr": 0.001,
            "T": 6,
            "K": 30,
            "b": 0.0002,
            "max_sigma": 0.999,
            "alpha": 0.1,
        }
    diffusion(config)

@cli.command()
@click.option("--local-dir", type=str, default="diffusion-molecule")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=1)
@click.option("--num-parallel", type=int, default=1)
@click.option("--num-samples", type=int, default=1)
def molecule(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on MoleculeActivity-v0
    """

    # Final Version

    from design_baselines.diffusion import diffusion

    ray.init(
        num_cpus=cpus,
        num_gpus=gpus,
        include_dashboard=False,
        temp_dir=os.path.expanduser(f"~/tmp_{randint(0, 1000000)}"),
    )
    tune.run(
        diffusion,
        config={
            "logging_dir": "data",
            "task": "MoleculeActivity-v0",
            "task_kwargs": {"split_percentile": 80},
            "is_discrete": True,
            "normalize_ys": True,
            "normalize_xs": False,
            "discrete_smoothing": 0.6,
            "val_size": 500,
            "batch_size": 128,
            "epochs": 500,
            "activations": ["leaky_relu", "leaky_relu"],
            "hidden_size": 2048,
            "initial_max_std": 0.2,
            "initial_min_std": 0.1,
            "forward_model_lr": 0.001,
            "initial_alpha": 1.0,
            "alpha_lr": 0.05,
            "target_conservatism": 0.05,
            "negatives_fraction": 0.5,
            "lookahead_steps": 1,
            "lookahead_backprop": True,
            "solver_lr": 0.01,
            "solver_interval": 1,
            "solver_warmup": 50,
            "solver_steps": 1,
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={"cpu": cpus // num_parallel, "gpu": gpus / num_parallel - 0.01},
    )


@cli.command()
@click.option("--local-dir", type=str, default="diffusion-gfp")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=1)
@click.option("--num-parallel", type=int, default=1)
@click.option("--num-samples", type=int, default=1)
def gfp(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on GFP-v0
    """

    # Final Version

    from design_baselines.diffusion import diffusion

    ray.init(
        num_cpus=cpus,
        num_gpus=gpus,
        include_dashboard=False,
        temp_dir=os.path.expanduser(f"~/tmp_{randint(0, 1000000)}"),
    )
    tune.run(
        diffusion,
        config={
            "logging_dir": "data",
            "task": "GFP-v0",
            "task_kwargs": {"seed": tune.randint(1000)},
            "is_discrete": True,
            "normalize_ys": True,
            "normalize_xs": False,
            "discrete_smoothing": 0.6,
            "val_size": 500,
            "batch_size": 128,
            "epochs": 500,
            "activations": ["leaky_relu", "leaky_relu"],
            "hidden_size": 2048,
            "initial_max_std": 0.2,
            "initial_min_std": 0.1,
            "forward_model_lr": 0.001,
            "initial_alpha": 1.0,
            "alpha_lr": 0.05,
            "target_conservatism": 0.05,
            "negatives_fraction": 0.5,
            "lookahead_steps": 1,
            "lookahead_backprop": True,
            "solver_lr": 0.01,
            "solver_interval": 1,
            "solver_warmup": 50,
            "solver_steps": 1,
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={"cpu": cpus // num_parallel, "gpu": gpus / num_parallel - 0.01},
    )


@cli.command()
@click.option("--local-dir", type=str, default="diffusion-dkitty")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=1)
@click.option("--num-parallel", type=int, default=1)
@click.option("--num-samples", type=int, default=1)
def dkitty(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on DKittyMorphology-v0
    """

    # Final Version

    from design_baselines.diffusion import diffusion

    ray.init(
        num_cpus=cpus,
        num_gpus=gpus,
        include_dashboard=False,
        temp_dir=os.path.expanduser(f"~/tmp_{randint(0, 1000000)}"),
    )
    tune.run(
        diffusion,
        config={
            "logging_dir": "data",
            "task": "DKittyMorphology-v0",
            "task_kwargs": {"split_percentile": 40, "num_parallel": 2},
            "is_discrete": False,
            "normalize_ys": True,
            "normalize_xs": True,
            "continuous_noise_std": 0.0,
            "val_size": 500,
            "batch_size": 128,
            "epochs": 500,
            "activations": ["leaky_relu", "leaky_relu"],
            "hidden_size": 2048,
            "initial_max_std": 0.2,
            "initial_min_std": 0.1,
            "forward_model_lr": 0.001,
            "initial_alpha": 1.0,
            "alpha_lr": 0.05,
            "target_conservatism": 0.05,
            "negatives_fraction": 0.5,
            "lookahead_steps": 1,
            "lookahead_backprop": True,
            "solver_lr": 0.01,
            "solver_interval": 1,
            "solver_warmup": 50,
            "solver_steps": 1,
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={"cpu": cpus // num_parallel, "gpu": gpus / num_parallel - 0.01},
    )


@cli.command()
@click.option("--local-dir", type=str, default="diffusion-ant")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=1)
@click.option("--num-parallel", type=int, default=1)
@click.option("--num-samples", type=int, default=1)
def ant(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on AntMorphology-v0
    """

    # Final Version

    from design_baselines.diffusion import diffusion

    ray.init(
        num_cpus=cpus,
        num_gpus=gpus,
        include_dashboard=False,
        temp_dir=os.path.expanduser(f"~/tmp_{randint(0, 1000000)}"),
    )
    tune.run(
        diffusion,
        config={
            "logging_dir": "data",
            "task": "AntMorphology-v0",
            "task_kwargs": {"split_percentile": 20, "num_parallel": 2},
            "is_discrete": False,
            "normalize_ys": True,
            "normalize_xs": True,
            "continuous_noise_std": 0.0,
            "val_size": 500,
            "batch_size": 128,
            "epochs": 500,
            "activations": ["leaky_relu", "leaky_relu"],
            "hidden_size": 2048,
            "initial_max_std": 0.2,
            "initial_min_std": 0.1,
            "forward_model_lr": 0.001,
            "initial_alpha": 1.0,
            "alpha_lr": 0.05,
            "target_conservatism": 0.05,
            "negatives_fraction": 0.5,
            "lookahead_steps": 1,
            "lookahead_backprop": True,
            "solver_lr": 0.01,
            "solver_interval": 1,
            "solver_warmup": 50,
            "solver_steps": 1,
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={"cpu": cpus // num_parallel, "gpu": gpus / num_parallel - 0.01},
    )


@cli.command()
@click.option("--local-dir", type=str, default="diffusion-hopper")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=1)
@click.option("--num-parallel", type=int, default=1)
@click.option("--num-samples", type=int, default=1)
def hopper(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on HopperController-v0
    """

    # Final Version

    from design_baselines.diffusion import diffusion

    ray.init(
        num_cpus=cpus,
        num_gpus=gpus,
        include_dashboard=False,
        temp_dir=os.path.expanduser(f"~/tmp_{randint(0, 1000000)}"),
    )
    tune.run(
        diffusion,
        config={
            "logging_dir": "data",
            "task": "HopperController-v0",
            "task_kwargs": {},
            "is_discrete": False,
            "normalize_ys": True,
            "normalize_xs": True,
            "continuous_noise_std": 0.0,
            "val_size": 200,
            "batch_size": 128,
            "epochs": 500,
            "activations": ["leaky_relu", "leaky_relu"],
            "hidden_size": 2048,
            "initial_max_std": 0.2,
            "initial_min_std": 0.1,
            "forward_model_lr": 0.001,
            "T": 6,
            "K": 30,
            "b": 0.001,
            "max_sigma": 0.999,
            "alpha": tune.grid_search([1.0, 0.1]),
            "target_scale": tune.grid_search([20.0, 50.0]),
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={"cpu": cpus // num_parallel, "gpu": gpus / num_parallel - 0.01},
    )


@cli.command()
@click.option("--local-dir", type=str, default="diffusion-superconductor")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=1)
@click.option("--num-parallel", type=int, default=1)
@click.option("--num-samples", type=int, default=1)
def superconductor(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on Superconductor-v0
    """

    # Final Version

    from design_baselines.diffusion import diffusion

    ray.init(
        num_cpus=cpus,
        num_gpus=gpus,
        include_dashboard=False,
        temp_dir=os.path.expanduser(f"~/tmp_{randint(0, 1000000)}"),
    )
    tune.run(
        diffusion,
        config={
            "logging_dir": "data",
            "task": "Superconductor-v0",
            "task_kwargs": {},
            "is_discrete": False,
            "normalize_ys": True,
            "normalize_xs": True,
            "continuous_noise_std": 0.2,
            "val_size": 500,
            "batch_size": 128,
            "epochs": 500,
            "activations": ["leaky_relu", "leaky_relu"],
            "hidden_size": 2048,
            "initial_max_std": 0.2,
            "initial_min_std": 0.1,
            "forward_model_lr": 0.001,
            "initial_alpha": 1.0,
            "alpha_lr": 0.05,
            "target_conservatism": 0.05,
            "negatives_fraction": 0.5,
            "lookahead_steps": 1,
            "lookahead_backprop": True,
            "solver_lr": 0.01,
            "solver_interval": 1,
            "solver_warmup": 50,
            "solver_steps": 1,
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={"cpu": cpus // num_parallel, "gpu": gpus / num_parallel - 0.01},
    )