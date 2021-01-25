from design_baselines.data import StaticGraphTask
from design_baselines.logger import Logger
from design_baselines.utils import spearman
from design_baselines.utils import soft_noise
from design_baselines.diffusion.trainers import DiffusionRecoveryLikelihood
from design_baselines.diffusion.nets import ForwardModel
from collections import defaultdict
import tensorflow as tf
import numpy as np
import glob


def diffusion(config):
    # create the training task and logger
    logger = Logger(config["logging_dir"])
    task = StaticGraphTask(config["task"], **config["task_kwargs"])

    # save the initial dataset statistics for safe keeping
    x = task.x
    y = task.y
    if config["normalize_ys"]:
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

    if config["normalize_xs"] and not config["is_discrete"]:
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

    #config["sgld_lr"] *= np.sqrt(np.prod(x.shape[1:]))
    solution_shape = [config["batch_size"]] + list(x.shape[1:])
    # make a neural network to predict scores
    forward_model = ForwardModel(
        task.input_shape,
        activations=config["activations"],
        hidden=config["hidden_size"],
    )

    trainer = DiffusionRecoveryLikelihood(
        forward_model=forward_model,
        forward_model_opt=tf.keras.optimizers.Adam,
        forward_model_lr=config["forward_model_lr"],
        target_scale=config["target_scale"],
        alpha=config["alpha"],
        T=config["T"],
        K=config["K"],
        b=config["b"],
        max_sigma=config["max_sigma"],
        is_discrete=config["is_discrete"],
        continuous_noise_std=config.get("continuous_noise_std", 0.0),
        discrete_smoothing=config.get("discrete_smoothing", 0.6),
        )

    # create a data set
    train_data, validate_data = task.build(
        x=x, y=y, batch_size=config["batch_size"], val_size=config["val_size"]
    )

    step = 0
    # train model for many epochs
    for e in range(config["epochs"]):
        for x, y in train_data:
            statistics = trainer.train_step(x, y)
            step += 1
            for name in statistics.keys():
                logger.record(name, tf.concat(statistics[name], axis=0), step)

        # evaluate the sampled designs
        solution = tf.random.normal(solution_shape)
        solution = trainer.run_markovchain(solution)
        score = task.score(solution * st_x + mu_x)
        logger.record("score", score, e, percentile=True)

        energy = forward_model(tf.math.softmax(solution) if config["is_discrete"] else solution)
        logger.record("energy", energy / config["target_scale"], e, percentile=True)

