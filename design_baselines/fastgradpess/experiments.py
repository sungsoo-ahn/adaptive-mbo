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


@cli.command()
@click.option("--local-dir", type=str, default="fastgradpess-gfp")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=1)
@click.option("--num-parallel", type=int, default=1)
@click.option("--num-samples", type=int, default=1)
def gfp(local_dir, cpus, gpus, num_parallel, num_samples):
    from design_baselines.fastgradpess import fastgradpess

    ray.init(
        num_cpus=cpus, num_gpus=gpus, temp_dir=os.path.expanduser(f"~/tmp_{randint(0, 1000000)}"),
    )
    tune.run(
        fastgradpess,
        config={
            "logging_dir": "data",
            "task": "GFP-v0",
            "task_kwargs": {"seed": tune.randint(1000)},
            "is_discrete": True,
            "normalize_ys": True,
            "normalize_xs": False,
            "discrete_smoothing": tune.grid_search([0.6, 0.9]),
            "continuous_noise_std": 0.0,
            "val_size": 200,
            "batch_size": 128,
            "updates": 10000,
            "warmup_epochs": 200,
            "steps_per_update": 1000,
            "hidden_size": 256,
            "model_lr": 1e-3,
            "sol_x_samples": 128,
            "sol_x_lr": 1e-2,
            "coef_pessimism": tune.grid_search([1e0, 1e1]),
            "coef_stddev": 0.0,
            "score_freq": 100,
            "model_class": "doublehead",
            "ema_rate": tune.grid_search([0.0, 0.99])
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={"cpu": cpus // num_parallel, "gpu": gpus / num_parallel},
    )


@cli.command()
@click.option("--local-dir", type=str, default="fastgradpess-molecule")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=1)
@click.option("--num-parallel", type=int, default=1)
@click.option("--num-samples", type=int, default=1)
def molecule(local_dir, cpus, gpus, num_parallel, num_samples):
    from design_baselines.fastgradpess import fastgradpess

    ray.init(
        num_cpus=cpus, num_gpus=gpus, temp_dir=os.path.expanduser(f"~/tmp_{randint(0, 1000000)}"),
    )
    tune.run(
        fastgradpess,
        config={
            "logging_dir": "data",
            "task": "MoleculeActivity-v0",
            "task_kwargs": {"split_percentile": 80},
            "is_discrete": True,
            "normalize_ys": True,
            "normalize_xs": False,
            "discrete_smoothing": 0.6,
            "continuous_noise_std": 0.0,
            "val_size": 500,
            "batch_size": 128,
            "updates": 10000,
            "warmup_epochs": 200,
            "steps_per_update": 1000,
            "hidden_size": 256,
            "model_lr": 0.001,
            "sol_x_samples": 128,
            "sol_x_lr": 1e-1,
            "coef_pessimism": 1e1,
            "coef_stddev": 0.0,
            "score_freq": 100,
            "model_class": "doublehead",
            "ema_rate": 0.0,
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={"cpu": cpus // num_parallel, "gpu": gpus / num_parallel},
    )


@cli.command()
@click.option("--local-dir", type=str, default="fastgradpess-superconductor")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=1)
@click.option("--num-parallel", type=int, default=1)
@click.option("--num-samples", type=int, default=1)
def superconductor(local_dir, cpus, gpus, num_parallel, num_samples):
    from design_baselines.fastgradpess import fastgradpess

    ray.init(
        num_cpus=cpus, num_gpus=gpus, temp_dir=os.path.expanduser(f"~/tmp_{randint(0, 1000000)}"),
    )
    tune.run(
        fastgradpess,
        config={
            "logging_dir": "data",
            "task": "Superconductor-v0",
            "task_kwargs": {"split_percentile": 80},
            "is_discrete": False,
            "normalize_ys": True,
            "normalize_xs": True,
            "continuous_noise_std": 0.2,
            "val_size": 500,
            "batch_size": 128,
            "updates": 10000,
            "warmup_epochs": 200,
            "steps_per_update": 50,
            "hidden_size": 256,
            "model_lr": 0.001,
            "sol_x_samples": 128,
            "sol_x_lr": 1e-1,
            "sol_x_eps": 1e-3,
            "coef_pessimism": 1e3,
            "coef_stddev": 0.0,
            "score_freq": 100,
            "model_class": "doublehead",
            "ema_rate": 0.995,
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={"cpu": cpus // num_parallel, "gpu": gpus / num_parallel},
    )


@cli.command()
@click.option("--local-dir", type=str, default="fastgradpess-dkitty")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=1)
@click.option("--num-parallel", type=int, default=1)
@click.option("--num-samples", type=int, default=1)
def dkitty(local_dir, cpus, gpus, num_parallel, num_samples):
    from design_baselines.fastgradpess import fastgradpess

    ray.init(
        num_cpus=cpus, num_gpus=gpus, temp_dir=os.path.expanduser(f"~/tmp_{randint(0, 1000000)}"),
    )
    tune.run(
        fastgradpess,
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
            "updates": 10000,
            "warmup_epochs": 200,
            "steps_per_update": 1000,
            "hidden_size": 256,
            "model_lr": 0.001,
            "sol_x_samples": 128,
            "sol_x_lr": 1e-3,
            "coef_pessimism": 1.0,
            "coef_stddev": tune.grid_search([0.0, 1.0]),
            "score_freq": 1000,
            "model_class": "doublehead",
            "ema_rate": tune.grid_search([0.0, 0.995]),
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={"cpu": cpus // num_parallel, "gpu": gpus / num_parallel},
    )


@cli.command()
@click.option("--local-dir", type=str, default="fastgradpess-ant")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=1)
@click.option("--num-parallel", type=int, default=1)
@click.option("--num-samples", type=int, default=1)
def ant(local_dir, cpus, gpus, num_parallel, num_samples):
    from design_baselines.fastgradpess import fastgradpess

    ray.init(
        num_cpus=cpus, num_gpus=gpus, temp_dir=os.path.expanduser(f"~/tmp_{randint(0, 1000000)}"),
    )
    tune.run(
        fastgradpess,
        config={
            "logging_dir": "data",
            "task": "AntMorphology-v0",
            "task_kwargs": {}, #{"split_percentile": 20, "num_parallel": 2},
            "is_discrete": False,
            "normalize_ys": True,
            "normalize_xs": True,
            "continuous_noise_std": 0.0,
            "val_size": 200,
            "batch_size": 128,
            "updates": 10000,
            "warmup_epochs": 200,
            "steps_per_update": 1000,
            "hidden_size": 256,
            "model_lr": 0.001,
            "sol_x_samples": 128,
            "sol_x_lr": 1e-3,
            "coef_pessimism": 1e0,
            "coef_stddev": 0.0,
            "score_freq": 1000,
            "ema_rate": 0.0,
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={"cpu": cpus // num_parallel, "gpu": gpus / num_parallel},
    )


@cli.command()
@click.option("--local-dir", type=str, default="fastgradpess-hopper")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=1)
@click.option("--num-parallel", type=int, default=1)
@click.option("--num-samples", type=int, default=1)
def hopper(local_dir, cpus, gpus, num_parallel, num_samples):
    from design_baselines.fastgradpess import fastgradpess

    ray.init(
        num_cpus=cpus, num_gpus=gpus, temp_dir=os.path.expanduser(f"~/tmp_{randint(0, 1000000)}"),
    )
    tune.run(
        fastgradpess,
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
            "updates": 10000,
            "warmup_epochs": 200,
            "steps_per_update": 1000,
            "hidden_size": 256,
            "model_lr": 0.001,
            "sol_x_samples": 128,
            "sol_x_lr": 1e-3,
            "coef_pessimism": 1e1,
            "coef_stddev": 0.0,
            "score_freq": 100,
            "ema_rate": 0.0,
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={"cpu": cpus // num_parallel, "gpu": gpus / num_parallel},
    )

