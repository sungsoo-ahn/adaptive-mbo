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
@click.option("--local-dir", type=str, default="rollout-molecule")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=1)
@click.option("--num-parallel", type=int, default=1)
@click.option("--num-samples", type=int, default=1)
def molecule(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on AntMorphology-v0
    """

    # Final Version

    from design_baselines.rollout import rollout

    ray.init(
        num_cpus=cpus, num_gpus=gpus, include_dashboard=False, temp_dir=os.path.expanduser("~/tmp")
    )
    tune.run(
        rollout,
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
            "epochs": 100,
            "activations": ["leaky_relu", "leaky_relu"],
            "hidden_size": 2048,
            "initial_max_std": 0.2,
            "initial_min_std": 0.1,
            "forward_model_lr": 0.001,
            "solver_samples": 128,
            "outer_step_size": 0.01,
            "outer_steps": 200,
            "inner_step_size": 0.0001,
            "inner_steps": 1,
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={"cpu": cpus // num_parallel, "gpu": gpus / num_parallel - 0.01},
    )


@cli.command()
@click.option("--local-dir", type=str, default="rollout-gfp")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=1)
@click.option("--num-parallel", type=int, default=1)
@click.option("--num-samples", type=int, default=1)
def gfp(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on AntMorphology-v0
    """

    # Final Version

    from design_baselines.rollout import rollout

    ray.init(
        num_cpus=cpus, num_gpus=gpus, include_dashboard=False, temp_dir=os.path.expanduser("~/tmp")
    )
    tune.run(
        rollout,
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
            "epochs": 100,
            "activations": ["leaky_relu", "leaky_relu"],
            "hidden_size": 2048,
            "initial_max_std": 0.2,
            "initial_min_std": 0.1,
            "forward_model_lr": 0.001,
            "solver_samples": 128,
            "outer_step_size": 0.01,
            "outer_steps": 200,
            "inner_step_size": tune.grid_search([1e-2, 1e-3]),
            "inner_steps": 5,
            "min_fim": tune.grid_search([1.0, 0.0]),
            "ewc_coef": tune.grid_search([100.0, 1.0]),
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={"cpu": cpus // num_parallel, "gpu": gpus / num_parallel - 0.01},
    )


@cli.command()
@click.option("--local-dir", type=str, default="rollout-gfp-v1")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=1)
@click.option("--num-parallel", type=int, default=1)
@click.option("--num-samples", type=int, default=1)
def gfp_v1(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on GFP-v1
    """

    # Final Version

    from design_baselines.rollout import rollout

    ray.init(
        num_cpus=cpus, num_gpus=gpus, include_dashboard=False, temp_dir=os.path.expanduser("~/tmp")
    )
    tune.run(
        rollout,
        config={
            "logging_dir": "data",
            "task": "GFP-v1",
            "task_kwargs": {"split_percentile": 20},
            "is_discrete": True,
            "normalize_ys": True,
            "normalize_xs": False,
            "discrete_smoothing": 0.6,
            "val_size": 200,
            "batch_size": 128,
            "epochs": 100,
            "activations": ["leaky_relu", "leaky_relu"],
            "hidden_size": 2048,
            "initial_max_std": 0.2,
            "initial_min_std": 0.1,
            "forward_model_lr": 0.001,
            "solver_samples": 128,
            "outer_step_size": 0.01,
            "outer_steps": 200,
            "inner_step_size": 0.0001,
            "inner_steps": 1,
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={"cpu": cpus // num_parallel, "gpu": gpus / num_parallel - 0.01},
    )


@cli.command()
@click.option("--local-dir", type=str, default="rollout-dkitty")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=1)
@click.option("--num-parallel", type=int, default=1)
@click.option("--num-samples", type=int, default=1)
def dkitty(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on DKittyMorphology-v0
    """

    # Final Version

    from design_baselines.rollout import rollout

    ray.init(
        num_cpus=cpus, num_gpus=gpus, include_dashboard=False, temp_dir=os.path.expanduser("~/tmp")
    )
    tune.run(
        rollout,
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
            "epochs": 100,
            "activations": ["leaky_relu", "leaky_relu"],
            "hidden_size": 2048,
            "initial_max_std": 0.2,
            "initial_min_std": 0.1,
            "forward_model_lr": 0.001,
            "solver_samples": 128,
            "outer_step_size": 0.01,
            "outer_steps": 200,
            "inner_step_size": 0.0001,
            "inner_steps": 1,
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={"cpu": cpus // num_parallel, "gpu": gpus / num_parallel - 0.01},
    )


@cli.command()
@click.option("--local-dir", type=str, default="rollout-ant")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=1)
@click.option("--num-parallel", type=int, default=1)
@click.option("--num-samples", type=int, default=1)
def ant(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on AntMorphology-v0
    """

    # Final Version

    from design_baselines.rollout import rollout

    ray.init(
        num_cpus=cpus, num_gpus=gpus, include_dashboard=False, temp_dir=os.path.expanduser("~/tmp")
    )
    tune.run(
        rollout,
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
            "hidden_size": 2048,
            "initial_max_std": 0.2,
            "initial_min_std": 0.1,
            "forward_model_lr": 0.001,
            "solver_samples": 128,
            "outer_step_size": 0.01,
            "outer_steps": 200,
            "inner_step_size": 0.0001,
            "inner_steps": 1,
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={"cpu": cpus // num_parallel, "gpu": gpus / num_parallel - 0.01},
    )


@cli.command()
@click.option("--local-dir", type=str, default="rollout-hopper")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=1)
@click.option("--num-parallel", type=int, default=1)
@click.option("--num-samples", type=int, default=1)
def hopper(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on HopperController-v0
    """

    # Final Version

    from design_baselines.rollout import rollout

    ray.init(
        num_cpus=cpus, num_gpus=gpus, include_dashboard=False, temp_dir=os.path.expanduser("~/tmp")
    )
    tune.run(
        rollout,
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
            "hidden_size": 2048,
            "initial_max_std": 0.2,
            "initial_min_std": 0.1,
            "forward_model_lr": 0.001,
            "solver_samples": 128,
            "outer_step_size": 0.01,
            "outer_steps": 200,
            "inner_step_size": 0.0001,
            "inner_steps": 1,
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={"cpu": cpus // num_parallel, "gpu": gpus / num_parallel - 0.01},
    )


@cli.command()
@click.option("--local-dir", type=str, default="rollout-superconductor")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=1)
@click.option("--num-parallel", type=int, default=1)
@click.option("--num-samples", type=int, default=1)
def superconductor(local_dir, cpus, gpus, num_parallel, num_samples):
    """Evaluate Conservative Score Models on Superconductor-v0
    """

    # Final Version

    from design_baselines.rollout import rollout

    ray.init(
        num_cpus=cpus, num_gpus=gpus, include_dashboard=False, temp_dir=os.path.expanduser("~/tmp")
    )
    tune.run(
        rollout,
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
            "epochs": 100,
            "activations": ["leaky_relu", "leaky_relu"],
            "hidden_size": 2048,
            "initial_max_std": 0.2,
            "initial_min_std": 0.1,
            "forward_model_lr": 0.001,
            "solver_samples": 128,
            "outer_step_size": 0.01,
            "outer_steps": 200,
            "inner_step_size": 0.0001,
            "inner_steps": 1,
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={"cpu": cpus // num_parallel, "gpu": gpus / num_parallel - 0.01},
    )