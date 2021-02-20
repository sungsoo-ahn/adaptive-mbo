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
@click.option("--local-dir", type=str, default="energy-gfp")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=1)
@click.option("--num-parallel", type=int, default=1)
@click.option("--num-samples", type=int, default=1)
def gfp(local_dir, cpus, gpus, num_parallel, num_samples):
    from design_baselines.energy import energy

    ray.init(
        num_cpus=cpus,
        num_gpus=gpus,
        include_dashboard=False,
        temp_dir=os.path.expanduser(f"~/tmp_{randint(0, 1000000)}"),
    )
    tune.run(
        energy,
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
            "pretrain_epochs": 200,
            "epochs": 100,
            "log_freq": 0,
            "hidden_size": 8192,
            "model_lr": 0.001,
            "buffer_size": 4096,
            "buffer_update_size": 512,
            "buffer_update_freq": 1000,
            "sgld_lr": 1e-2,
            "sgld_noise_penalty": tune.grid_search([1.0, 0.1]),
            "pcd_steps": 10,
            "warmup_steps": 20000,
            "reg_coef": 10.0,
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={"cpu": cpus // num_parallel, "gpu": gpus / num_parallel - 0.01},
    )

@cli.command()
@click.option("--local-dir", type=str, default="energy-molecule")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=1)
@click.option("--num-parallel", type=int, default=1)
@click.option("--num-samples", type=int, default=1)
def molecule(local_dir, cpus, gpus, num_parallel, num_samples):
    from design_baselines.energy import energy

    ray.init(
        num_cpus=cpus,
        num_gpus=gpus,
        include_dashboard=False,
        temp_dir=os.path.expanduser(f"~/tmp_{randint(0, 1000000)}"),
    )
    tune.run(
        energy,
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
            "epochs": 1000,
            "hidden_size": 256,
            "model_lr": 0.001,
            "buffer_size": 50000,
            "buffer_update_size": 1024,
            "buffer_update_freq": 1,
            "sgld_lr": 1e-2,
            "sgld_noise_penalty": 1.0,
            "mc_pcd_steps": 20,
            "reg_coef": 1e-3,
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={"cpu": cpus // num_parallel, "gpu": gpus / num_parallel - 0.01},
    )


@cli.command()
@click.option("--local-dir", type=str, default="energy-superconductor")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=1)
@click.option("--num-parallel", type=int, default=1)
@click.option("--num-samples", type=int, default=1)
def superconductor(local_dir, cpus, gpus, num_parallel, num_samples):
    from design_baselines.energy import energy

    ray.init(
        num_cpus=cpus,
        num_gpus=gpus,
        include_dashboard=False,
        temp_dir=os.path.expanduser(f"~/tmp_{randint(0, 1000000)}"),
    )
    tune.run(
        energy,
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
            "pretrain_epochs": 200,
            "warmups": 100,
            "epochs": 500,
            "hidden_size": 256,
            "model_lr": 0.001,
            "buffer_size": 4096,
            "reinit_freq": tune.grid_search([0.05, 0.2]),
            "sgld_lr": 0.01,
            "sgld_noise_penalty": tune.grid_search([0.1, 0.001]),
            "mc_pcd_steps": 20,
            "reg_coef": 1e0,
            "solver_freq": 50,
            "solver_samples": 128,
            "solver_steps": 5000,
            "solver_lr": 0.01,
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={"cpu": cpus // num_parallel, "gpu": gpus / num_parallel - 0.01},
    )

@cli.command()
@click.option("--local-dir", type=str, default="energy-dkitty")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=1)
@click.option("--num-parallel", type=int, default=1)
@click.option("--num-samples", type=int, default=1)
def dkitty(local_dir, cpus, gpus, num_parallel, num_samples):
    from design_baselines.energy import energy

    ray.init(
        num_cpus=cpus,
        num_gpus=gpus,
        include_dashboard=False,
        temp_dir=os.path.expanduser(f"~/tmp_{randint(0, 1000000)}"),
    )
    tune.run(
        energy,
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
@click.option("--local-dir", type=str, default="energy-ant")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=1)
@click.option("--num-parallel", type=int, default=1)
@click.option("--num-samples", type=int, default=1)
def ant(local_dir, cpus, gpus, num_parallel, num_samples):
    from design_baselines.energy import energy

    ray.init(
        num_cpus=cpus,
        num_gpus=gpus,
        include_dashboard=False,
        temp_dir=os.path.expanduser(f"~/tmp_{randint(0, 1000000)}"),
    )
    tune.run(
        energy,
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
@click.option("--local-dir", type=str, default="energy-hopper")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=1)
@click.option("--num-parallel", type=int, default=1)
@click.option("--num-samples", type=int, default=1)
def hopper(local_dir, cpus, gpus, num_parallel, num_samples):
    from design_baselines.energy import energy

    ray.init(
        num_cpus=cpus,
        num_gpus=gpus,
        include_dashboard=False,
        temp_dir=os.path.expanduser(f"~/tmp_{randint(0, 1000000)}"),
    )
    tune.run(
        energy,
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

