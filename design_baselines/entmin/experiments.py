from ray import tune
import click
import ray
import os


@click.group()
def cli():
    """A group of experiments for training Conservative Score Models
    and reproducing our ICLR 2021 results.
    """


#############


@cli.command()
@click.option("--local-dir", type=str, default="entmin-molecule")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=1)
@click.option("--num-parallel", type=int, default=1)
@click.option("--num-samples", type=int, default=1)
def molecule(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on AntMorphology-v0
    """

    # Final Version

    from design_baselines.entmin import entmin

    ray.init(
        num_cpus=cpus, num_gpus=gpus, include_dashboard=False, temp_dir=os.path.expanduser("~/tmp")
    )
    tune.run(
        entmin,
        config={
            "logging_dir": "data",
            "task": "MoleculeActivity-v0",
            "task_kwargs": {"split_percentile": 80},
            "is_discrete": True,
            "normalize_ys": True,
            "normalize_xs": False,
            "discrete_smoothing": 0.6,
            "val_size": 200,
            "batch_size": 128,
            "epochs": 10000, #20000,
            "warmup": 100,
            "hidden_size": 64,
            "initial_max_std": 0.2,
            "initial_min_std": 0.1,
            "forward_model_lr": 0.001,
            "solver_samples": 128,
            "alpha": tune.grid_search([1e0, 1e-1, 1e-2, 1e-3]),
            "x_lr": 1e-1,
            "update_freq": 1,
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={"cpu": cpus // num_parallel, "gpu": gpus / num_parallel - 0.01},
    )


@cli.command()
@click.option("--local-dir", type=str, default="entmin-gfp")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=1)
@click.option("--num-parallel", type=int, default=1)
@click.option("--num-samples", type=int, default=1)
def gfp(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on AntMorphology-v0
    """

    # Final Version

    from design_baselines.entmin import entmin

    ray.init(
        num_cpus=cpus, num_gpus=gpus, include_dashboard=False, temp_dir=os.path.expanduser("~/tmp")
    )
    tune.run(
        entmin,
        config={
            "logging_dir": "data",
            "task": "GFP-v0",
            "task_kwargs": {"seed": tune.randint(1000), "split_percentile": 100},
            "is_discrete": True,
            "normalize_ys": True,
            "normalize_xs": False,
            "discrete_smoothing": 0.6,
            "val_size": 200,
            "batch_size": 128,
            "epochs": 10000, #20000,
            "warmup": 100,
            "hidden_size": 64,
            "initial_max_std": 0.2,
            "initial_min_std": 0.1,
            "forward_model_lr": 0.001,
            "solver_samples": 128,
            "alpha": tune.grid_search([1e1, 1e0, 1e-1, 1e-2]),
            "x_lr": 1e-2,
            "update_freq": 1,
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={"cpu": cpus // num_parallel, "gpu": gpus / num_parallel - 0.01},
    )


@cli.command()
@click.option("--local-dir", type=str, default="entmin-dkitty")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=1)
@click.option("--num-parallel", type=int, default=1)
@click.option("--num-samples", type=int, default=1)
def dkitty(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on DKittyMorphology-v0
    """

    # Final Version

    from design_baselines.entmin import entmin

    ray.init(
        num_cpus=cpus, num_gpus=gpus, include_dashboard=False, temp_dir=os.path.expanduser("~/tmp")
    )
    tune.run(
        entmin,
        config={
            "logging_dir": "data",
            "task": "DKittyMorphology-v0",
            "task_kwargs": {"split_percentile": 40, "num_parallel": 2},
            "is_discrete": False,
            "normalize_ys": True,
            "normalize_xs": True,
            "continuous_noise_std": 0.0,
            "val_size": 200,
            "batch_size": 128,
            "epochs": 50,
            "activations": ["leaky_relu", "leaky_relu"],
            "hidden_size": 256,
            "initial_max_std": 0.2,
            "initial_min_std": 0.1,
            "forward_model_lr": 0.001,
            "solver_samples": 128,
            "outer_step_size": 0.001,
            "outer_steps": 2000,
            "inner_step_size": tune.grid_search([1e-4, 1e-5]),
            "inner_steps": tune.grid_search([1, 2]),
            "log_freq": 2000,
            "num_models": 1,
            "optimizer": "adam",
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={"cpu": cpus // num_parallel, "gpu": gpus / num_parallel - 0.01},
    )


@cli.command()
@click.option("--local-dir", type=str, default="entmin-ant")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=1)
@click.option("--num-parallel", type=int, default=1)
@click.option("--num-samples", type=int, default=1)
def ant(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on AntMorphology-v0
    """

    # Final Version

    from design_baselines.entmin import entmin

    ray.init(
        num_cpus=cpus, num_gpus=gpus, include_dashboard=False, temp_dir=os.path.expanduser("~/tmp")
    )
    tune.run(
        entmin,
        config={
            "logging_dir": "data",
            "task": "AntMorphology-v0",
            "task_kwargs": {"split_percentile": 20, "num_parallel": 2},
            "is_discrete": False,
            "normalize_ys": True,
            "normalize_xs": True,
            "continuous_noise_std": 0.0,
            "val_size": 200,
            "batch_size": 128,
            "epochs": 100,
            "activations": ["leaky_relu", "leaky_relu"],
            "hidden_size": 256,
            "initial_max_std": 0.2,
            "initial_min_std": 0.1,
            "forward_model_lr": 0.001,
            "solver_samples": 128,
            "outer_step_size": 0.001,
            "outer_steps": 2000,
            "inner_step_size": tune.grid_search([1e-4, 1e-5]),
            "inner_steps": tune.grid_search([1, 2]),
            "log_freq": 2000,
            "num_models": 1,
            "optimizer": "adam",
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={"cpu": cpus // num_parallel, "gpu": gpus / num_parallel - 0.01},
    )


@cli.command()
@click.option("--local-dir", type=str, default="entmin-hopper")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=1)
@click.option("--num-parallel", type=int, default=1)
@click.option("--num-samples", type=int, default=1)
def hopper(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on HopperController-v0
    """

    # Final Version

    from design_baselines.entmin import entmin

    ray.init(
        num_cpus=cpus, num_gpus=gpus, include_dashboard=False, temp_dir=os.path.expanduser("~/tmp")
    )
    tune.run(
        entmin,
        config={
            "logging_dir": "data",
            "task": "HopperController-v0",
            "task_kwargs": {"split_percentile": 100},
            "is_discrete": False,
            "normalize_ys": True,
            "normalize_xs": True,
            "continuous_noise_std": 0.0,
            "val_size": 200,
            "batch_size": 128,
            "epochs": 100,
            "activations": ["leaky_relu", "leaky_relu"],
            "hidden_size": 256,
            "initial_max_std": 0.2,
            "initial_min_std": 0.1,
            "forward_model_lr": 0.001,
            "solver_samples": 128,
            "outer_step_size": 0.001,
            "outer_steps": 2000,
            "inner_step_size": tune.grid_search([1e-4, 1e-5]),
            "inner_steps": tune.grid_search([1, 2]),
            "log_freq": 2000,
            "num_models": 1,
            "optimizer": "adam",
            },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={"cpu": cpus // num_parallel, "gpu": gpus / num_parallel - 0.01},
    )


@cli.command()
@click.option("--local-dir", type=str, default="entmin-superconductor")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=1)
@click.option("--num-parallel", type=int, default=1)
@click.option("--num-samples", type=int, default=1)
def superconductor(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on Superconductor-v0
    """

    # Final Version

    from design_baselines.entmin import entmin

    ray.init(
        num_cpus=cpus, num_gpus=gpus, include_dashboard=False, temp_dir=os.path.expanduser("~/tmp")
    )
    tune.run(
        entmin,
        config={
            "logging_dir": "data",
            "task": "Superconductor-v0",
            "task_kwargs": {"split_percentile": 80},
            "is_discrete": False,
            "normalize_ys": True,
            "normalize_xs": True,
            "continuous_noise_std": 0.2,
            "val_size": 200,
            "batch_size": 128,
            "epochs": 10000, #20000,
            "warmup": 100,
            "hidden_size": 64,
            "initial_max_std": 0.2,
            "initial_min_std": 0.1,
            "forward_model_lr": 0.001,
            "solver_samples": 128,
            "alpha": 1e-3,
            "x_lr": 1e-1,
            "update_freq": 1,
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={"cpu": cpus // num_parallel, "gpu": gpus / num_parallel - 0.01},
    )