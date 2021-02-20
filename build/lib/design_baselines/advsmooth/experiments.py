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
@click.option("--local-dir", type=str, default="advsmooth-gfp")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=1)
@click.option("--num-parallel", type=int, default=1)
@click.option("--num-samples", type=int, default=1)
def gfp(local_dir, cpus, gpus, num_parallel, num_samples):
    from design_baselines.advsmooth import advsmooth

    ray.init(
        num_cpus=cpus, num_gpus=gpus, temp_dir=os.path.expanduser(f"~/tmp_{randint(0, 1000000)}"),
    )
    tune.run(
        advsmooth,
        config={
            "logging_dir": "data",
            "task": "GFP-v0",
            "task_kwargs": {"seed": tune.randint(1000)},
            "is_discrete": True,
            "normalize_ys": True,
            "normalize_xs": False,
            "discrete_smoothing": 0.8,
            "continuous_noise_std": 0.2,
            "val_size": 500,
            "batch_size": 128,
            "updates": 5000,
            "warmup_epochs": 100,
            "steps_per_update": 50,
            "hidden_size": 256,
            "model_lr": 0.001,
            "sol_x_samples": 128,
            "sol_x_lr": 0.01,
            "coef_advsmoothimism": tune.grid_search([1e-3, 1e-2]),
            "coef_smoothing": tune.grid_search([1e0, 1e1]),
            "score_freq": 100,
            "ema_rate": 0.999,
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={"cpu": cpus // num_parallel, "gpu": gpus / num_parallel - 0.01},
    )


@cli.command()
@click.option("--local-dir", type=str, default="advsmooth-molecule")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=1)
@click.option("--num-parallel", type=int, default=1)
@click.option("--num-samples", type=int, default=1)
def molecule(local_dir, cpus, gpus, num_parallel, num_samples):
    from design_baselines.advsmooth import advsmooth

    ray.init(
        num_cpus=cpus, num_gpus=gpus, temp_dir=os.path.expanduser(f"~/tmp_{randint(0, 1000000)}"),
    )
    tune.run(
        advsmooth,
        config={
            "logging_dir": "data",
            "task": "MoleculeActivity-v0",
            "task_kwargs": {"split_percentile": 80},
            "is_discrete": True,
            "normalize_ys": True,
            "normalize_xs": False,
            "discrete_smoothing": 0.6,
            "continuous_noise_std": 0.2,
            "val_size": 500,
            "batch_size": 128,
            "updates": 2000,
            "warmup_epochs": 100,
            "steps_per_update": 20,
            "hidden_size": 256,
            "model_lr": 0.001,
            "sol_x_samples": 128,
            "sol_x_lr": 0.1,
            "coef_advsmoothimism": 1e-3,
            "coef_smoothing": 1e2,
            "score_freq": 1000,
            "ema_rate": 0.999,
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={"cpu": cpus // num_parallel, "gpu": gpus / num_parallel - 0.01},
    )


@cli.command()
@click.option("--local-dir", type=str, default="advsmooth-superconductor")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=1)
@click.option("--num-parallel", type=int, default=1)
@click.option("--num-samples", type=int, default=1)
def superconductor(local_dir, cpus, gpus, num_parallel, num_samples):
    from design_baselines.advsmooth import advsmooth

    ray.init(
        num_cpus=cpus, num_gpus=gpus, temp_dir=os.path.expanduser(f"~/tmp_{randint(0, 1000000)}"),
    )
    tune.run(
        advsmooth,
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
            "updates": 2000,
            "warmup_epochs": 100,
            "steps_per_update": 20,
            "hidden_size": 256,
            "model_lr": 0.001,
            "sol_x_samples": 128,
            "sol_x_lr": 0.1,
            "adv_rate": tune.grid_search([0.01, 0.1]),
            "coef_smoothing": tune.grid_search([1e0, 1e2]),
            "score_freq": 1000,
            "ema_rate": 0.999,
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={"cpu": cpus // num_parallel, "gpu": gpus / num_parallel - 0.01},
    )


@cli.command()
@click.option("--local-dir", type=str, default="advsmooth-dkitty")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=1)
@click.option("--num-parallel", type=int, default=1)
@click.option("--num-samples", type=int, default=1)
def dkitty(local_dir, cpus, gpus, num_parallel, num_samples):
    from design_baselines.advsmooth import advsmooth

    ray.init(
        num_cpus=cpus, num_gpus=gpus, temp_dir=os.path.expanduser(f"~/tmp_{randint(0, 1000000)}"),
    )
    tune.run(
        advsmooth,
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
            "pretrain_epochs": 200,
            "epochs": 100,
            "log_freq": 0,
            "hidden_size": 8192,
            "model_lr": 0.001,
            "buffer_size": 4096,
            "buffer_update_size": 512,
            "buffer_update_freq": 1000,
            "sgld_lr": 1e-2,
            "sgld_noise_penalty": 0.001,
            "pcd_steps": 10,
            "warmup_steps": 20000,
            "reg_coef": 1e-3,
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={"cpu": cpus // num_parallel, "gpu": gpus / num_parallel - 0.01},
    )


@cli.command()
@click.option("--local-dir", type=str, default="advsmooth-ant")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=1)
@click.option("--num-parallel", type=int, default=1)
@click.option("--num-samples", type=int, default=1)
def ant(local_dir, cpus, gpus, num_parallel, num_samples):
    from design_baselines.advsmooth import advsmooth

    ray.init(
        num_cpus=cpus, num_gpus=gpus, temp_dir=os.path.expanduser(f"~/tmp_{randint(0, 1000000)}"),
    )
    tune.run(
        advsmooth,
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
            "pretrain_epochs": 200,
            "epochs": 100,
            "log_freq": 0,
            "hidden_size": 8192,
            "model_lr": 0.001,
            "buffer_size": 4096,
            "buffer_update_size": 512,
            "buffer_update_freq": 1000,
            "sgld_lr": 1e-2,
            "sgld_noise_penalty": 0.001,
            "pcd_steps": 10,
            "warmup_steps": 20000,
            "reg_coef": 1e-3,
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={"cpu": cpus // num_parallel, "gpu": gpus / num_parallel - 0.01},
    )


@cli.command()
@click.option("--local-dir", type=str, default="advsmooth-hopper")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=1)
@click.option("--num-parallel", type=int, default=1)
@click.option("--num-samples", type=int, default=1)
def hopper(local_dir, cpus, gpus, num_parallel, num_samples):
    from design_baselines.advsmooth import advsmooth

    ray.init(
        num_cpus=cpus, num_gpus=gpus, temp_dir=os.path.expanduser(f"~/tmp_{randint(0, 1000000)}"),
    )
    tune.run(
        advsmooth,
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
            "pretrain_epochs": 200,
            "epochs": 100,
            "log_freq": 0,
            "hidden_size": 8192,
            "model_lr": 0.001,
            "buffer_size": 4096,
            "buffer_update_size": 512,
            "buffer_update_freq": 1000,
            "sgld_lr": 1e-2,
            "sgld_noise_penalty": 0.001,
            "pcd_steps": 10,
            "warmup_steps": 20000,
            "reg_coef": 1e-3,
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={"cpu": cpus // num_parallel, "gpu": gpus / num_parallel - 0.01},
    )

