from design_baselines.data import StaticGraphTask
from design_baselines.logger import Logger
from design_baselines.utils import spearman
from design_baselines.utils import perturb
from design_baselines.energy.trainers import Trainer, Solver
from design_baselines.energy.buffers import Buffer
from design_baselines.energy.nets import ForwardModel
from collections import defaultdict
import tensorflow as tf
import numpy as np
from tensorflow_probability import distributions as tfpd


def normalize_dataset(x, y, normalize_xs, normalize_ys):
    if normalize_ys:
        # compute normalization statistics for the score
        mu_y = np.mean(y, axis=0, keepdims=True).astype(np.float32)
        y = y - mu_y
        st_y = np.std(y, axis=0, keepdims=True).astype(np.float32)
        st_y = np.where(np.equal(st_y, 0), 1, st_y)
        y = y / st_y
    else:
        # compute normalization statistics for the score
        mu_y = np.zeros_like(y[:1])
        st_y = np.ones_like(y[:1])

    if normalize_xs:
        # compute normalization statistics for the data vectors
        mu_x = np.mean(x, axis=0, keepdims=True).astype(np.float32)
        x = x - mu_x
        st_x = np.std(x, axis=0, keepdims=True).astype(np.float32)
        st_x = np.where(np.equal(st_x, 0), 1, st_x)
        x = x / st_x
    else:
        # compute normalization statistics for the score
        mu_x = np.zeros_like(x[:1])
        st_x = np.ones_like(x[:1])

    return (x, mu_x, st_x), (y, mu_y, st_y)


def energy(config):
    # create the training task and logger
    logger = Logger(config["logging_dir"])
    task = StaticGraphTask(config["task"], **config["task_kwargs"])
    (task_x, mu_x, st_x), (task_y, mu_y, st_y) = normalize_dataset(
        x=task.x,
        y=task.y,
        normalize_xs=config["normalize_xs"],
        normalize_ys=config["normalize_ys"],
    )
    model = ForwardModel(
        input_shape=task.input_shape,
        hidden=config["hidden_size"],
        is_discrete=config["is_discrete"],
    )
    model_opt = tf.keras.optimizers.Adam(learning_rate=config["model_lr"])
    prior = tfpd.Normal(loc=0, scale=1)
    buffer = Buffer(
        x=task.x,
        y=task.y,
        buffer_size=config["buffer_size"],
        reinit_freq=config["reinit_freq"],
        )
    trainer = Trainer(
        model=model,
        model_opt=model_opt,
        prior=prior,
        sgld_lr=config["sgld_lr"],
        sgld_noise_penalty=config["sgld_noise_penalty"],
        reg_coef=config["reg_coef"],
    )

    train_data, validate_data = task.build(
        x=task_x, y=task_y, batch_size=config["batch_size"], val_size=config["val_size"]
    )

    for epoch in range(config["epochs"]):
        statistics = defaultdict(list)
        for x, y in train_data:
            if epoch < config["warmups"]:
                train_statistics = trainer.warmup_step(x, y)
            else:
                x_neg = buffer.pop(num_samples=config["batch_size"])
                x_neg, mc_persistent_statistics = trainer.run_markovchains(
                    x_neg, steps=config["mc_pcd_steps"]
                    )
                buffer.add(x_neg)
                train_statistics = trainer.train_step(x, y, x_neg)
                for name, tsr in mc_persistent_statistics.items():
                    statistics[f"markovchain/persistent/{name}"].append(tsr)

            for name, tsr in train_statistics.items():
                statistics[f"train/{name}"].append(tsr)

        for name in statistics.keys():
            logger.record(f"{name}", tf.reduce_mean(tf.concat(statistics[name], axis=0)), epoch)

        if (epoch + 1) > config["warmups"] and (epoch + 1) % config["solver_freq"] == 0:
            for stddev_coef in [0.0, 0.1, 1.0]:
                sol = buffer.get_topk_x(num_samples=config["solver_samples"])
                sol = tf.Variable(sol)
                sol_optim = tf.keras.optimizers.SGD(learning_rate=config["solver_lr"])
                solver = Solver(
                    model=model, prior=prior, sol=sol, sol_optim=sol_optim, stddev_coef=stddev_coef
                    )

                for solver_step in range(config["solver_steps"]):
                    solver_statistics = solver.solve_step()
                    for name, tsr in solver_statistics.items():
                        logger.record(
                            f"solver_{epoch}/stddev_{stddev_coef:.1f}/{name}", tsr, solver_step
                            )

                inp = tf.math.softmax(solver.sol) if config["is_discrete"] else solver.sol
                score = task.score(inp * st_x + mu_x)
                logger.record(
                    f"solver_{epoch}/stddev_{stddev_coef:.1f}/score", score, 0, percentile=True
                    )