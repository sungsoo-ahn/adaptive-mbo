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
@click.option("--local-dir", type=str, default="ensemble-molecule")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=1)
@click.option("--num-parallel", type=int, default=1)
@click.option("--num-samples", type=int, default=1)
def molecule(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on AntMorphology-v0
    """

    # Final Version

    from design_baselines.ensemble import ensemble

    ray.init(
        num_cpus=cpus, num_gpus=gpus, include_dashboard=False, temp_dir=os.path.expanduser("~/tmp")
    )
    tune.run(
        ensemble,
        config={
            "logging_dir": "data",
            "task": "MoleculeActivity-v0",
            "task_kwargs": {"split_percentile": 80},
            "is_discrete": True,
            "normalize_ys": True,
            "normalize_xs": False,
            "continuous_noise_std": 0.2,
            "discrete_smoothing": 0.8,
            "val_size": 200,
            "batch_size": 128,
            "updates": 5000,
            "warmup": 5000,
            "update_freq": 50,
            "score_freq": 100,
            "hidden_size": 256,
            "initial_max_std": 0.2,
            "initial_min_std": 0.1,
            "model_lr": 0.001,
            "solver_samples": 128,
            "sol_x_optim": "adam",
            "sol_x_lr": 0.1,
            "smoothing_coef": 1e1,
            "mc_evals": 10,
            "ema_rate": 0.999,
            "num_models": 1,
            "ensemble_type": tune.grid_search(["gaussian", "bernoulli"]),
            "ensemble_rate": tune.grid_search([0.1, 0.2, 0.5]),
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={"cpu": cpus // num_parallel, "gpu": gpus / num_parallel - 0.01},
    )
    ray.shutdown()


@cli.command()
@click.option("--local-dir", type=str, default="ensemble-gfp")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=1)
@click.option("--num-parallel", type=int, default=1)
@click.option("--num-samples", type=int, default=1)
def gfp(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on AntMorphology-v0
    """

    # Final Version

    from design_baselines.ensemble import ensemble

    ray.init(
        num_cpus=cpus, num_gpus=gpus, include_dashboard=False, temp_dir=os.path.expanduser("~/tmp")
    )
    tune.run(
        ensemble,
        config={
            "logging_dir": "data",
            "task": "GFP-v0",
            "task_kwargs": {"split_percentile": 100},
            "is_discrete": True,
            "normalize_ys": True,
            "normalize_xs": False,
            "discrete_smoothing": 0.8,
            "val_size": 200,
            "batch_size": 128,
            "updates": 1000,
            "warmup": 5000,
            "update_freq": 50,
            "score_freq": 100,
            "hidden_size": 256,
            "initial_max_std": 0.2,
            "initial_min_std": 0.1,
            "model_lr": 1e-3,
            "solver_samples": 128,
            "sol_x_optim": "adam",
            "sol_x_lr": 0.01,
            "smoothing_coef": 1e2,
            "mc_evals": 1,
            "ema_rate": 0.999,
            "num_models": 1,
            "ensemble_type": tune.grid_search(["gaussian", "bernoulli"]),
            "ensemble_rate": tune.grid_search([0.1, 0.2, 0.5]),
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={"cpu": cpus // num_parallel, "gpu": gpus / num_parallel - 0.01},
    )
    ray.shutdown()


@cli.command()
@click.option("--local-dir", type=str, default="ensemble-superconductor")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=1)
@click.option("--num-parallel", type=int, default=1)
@click.option("--num-samples", type=int, default=1)
def superconductor(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on Superconductor-v0
    """

    # Final Version

    from design_baselines.ensemble import ensemble

    ray.init(
        num_cpus=cpus, num_gpus=gpus, include_dashboard=False, temp_dir=os.path.expanduser("~/tmp")
    )
    tune.run(
        ensemble,
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
            "epochs": 200,
            "model_wd": tune.grid_search([0.02, 0.05]),
            "model_lr": 0.001,
            "ensemble_size": 1,
            "hidden_size": 256,
            "initial_max_std": 0.2,
            "initial_min_std": 0.1,
            "solver_samples": 128,
            "updates": 5000,
            "sol_lr": 1e-1,
            "score_freq": 100,
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={"cpu": cpus // num_parallel, "gpu": gpus / num_parallel - 0.01},
    )
    ray.shutdown()


@cli.command()
@click.option("--local-dir", type=str, default="ensemble-dkitty")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=1)
@click.option("--num-parallel", type=int, default=1)
@click.option("--num-samples", type=int, default=1)
def dkitty(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on DKittyMorphology-v0
    """

    # Final Version

    from design_baselines.ensemble import ensemble

    ray.init(
        num_cpus=cpus, num_gpus=gpus, include_dashboard=False, temp_dir=os.path.expanduser("~/tmp")
    )
    tune.run(
        ensemble,
        config={
            "logging_dir": "data",
            "task": "DKittyMorphology-v0",
            "task_kwargs": {"split_percentile": 40, "num_parallel": 2},
            "is_discrete": False,
            "normalize_ys": True,
            "normalize_xs": True,
            "continuous_noise_std": 1.0,
            "val_size": 200,
            "batch_size": 128,
            "updates": 5000,
            "warmup": 5000,
            "update_freq": 50,
            "score_freq": 5000,
            "hidden_size": 256,
            "initial_max_std": 0.2,
            "initial_min_std": 0.1,
            "model_lr": 0.001,
            "solver_samples": 128,
            "sol_x_optim": "adam",
            "sol_x_lr": 0.001,
            "y_prior_mean": -0.5,
            "y_prior_std": 1.0,
            "poterior_reg_coef": 0.1,
            "ema_rate": 0.999,
            "num_models": 1,
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={"cpu": cpus // num_parallel, "gpu": gpus / num_parallel - 0.01},
    )
    ray.shutdown()


@cli.command()
@click.option("--local-dir", type=str, default="ensemble-ant")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=1)
@click.option("--num-parallel", type=int, default=1)
@click.option("--num-samples", type=int, default=1)
def ant(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on AntMorphology-v0
    """

    # Final Version

    from design_baselines.ensemble import ensemble

    ray.init(
        num_cpus=cpus, num_gpus=gpus, include_dashboard=False, temp_dir=os.path.expanduser("~/tmp")
    )
    tune.run(
        ensemble,
        config={
            "logging_dir": "data",
            "task": "AntMorphology-v0",
            "task_kwargs": {"split_percentile": 20, "num_parallel": 2},
            "is_discrete": False,
            "normalize_ys": True,
            "normalize_xs": True,
            "continuous_noise_std": 1.0,
            "val_size": 200,
            "batch_size": 128,
            "updates": 5000,
            "warmup": 5000,
            "update_freq": 50,
            "score_freq": 5000,
            "hidden_size": 256,
            "initial_max_std": 0.2,
            "initial_min_std": 0.1,
            "model_lr": 0.001,
            "solver_samples": 128,
            "sol_x_optim": "adam",
            "sol_x_lr": 0.001,
            "y_prior_mean": -0.5,
            "y_prior_std": 1.0,
            "poterior_reg_coef": 0.1,
            "ema_rate": 0.99,
            "num_models": 1,
        },
        num_samples=num_samples,
        local_dir=local_dir_,
        resources_per_trial={"cpu": cpus // num_parallel, "gpu": gpus / num_parallel - 0.01},
    )
    ray.shutdown()


@cli.command()
@click.option("--local-dir", type=str, default="ensemble-hopper")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=1)
@click.option("--num-parallel", type=int, default=1)
@click.option("--num-samples", type=int, default=1)
def hopper(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on HopperController-v0
    """

    # Final Version

    from design_baselines.ensemble import ensemble

    ray.init(
        num_cpus=cpus, num_gpus=gpus, include_dashboard=False, temp_dir=os.path.expanduser("~/tmp")
    )
    tune.run(
        ensemble,
        config={
            "logging_dir": "data",
            "task": "HopperController-v0",
            "task_kwargs": {"split_percentile": 100},
            "is_discrete": False,
            "normalize_ys": True,
            "normalize_xs": True,
            "continuous_noise_std": 1.0,
            "val_size": 200,
            "batch_size": 128,
            "updates": 5000,
            "warmup": 5000,
            "update_freq": 50,
            "score_freq": 5000,
            "hidden_size": 256,
            "initial_max_std": 0.2,
            "initial_min_std": 0.1,
            "model_lr": 0.001,
            "solver_samples": 128,
            "sol_x_optim": "adam",
            "sol_x_lr": 0.001,
            "y_prior_mean": -0.5,
            "y_prior_std": 1.0,
            "poterior_reg_coef": 0.1,
            "ema_rate": 0.999,
            "num_models": 1,
            },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={"cpu": cpus // num_parallel, "gpu": gpus / num_parallel - 0.01},
    )