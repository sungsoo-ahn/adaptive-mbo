from design_baselines.data import StaticGraphTask
from design_baselines.logger import Logger
from design_baselines.utils import spearman
from design_baselines.utils import soft_noise, cont_noise
from design_baselines.rebm.trainers import MaximumLikelihood
from design_baselines.rebm.nets import ForwardModel
from collections import defaultdict
import tensorflow as tf
import numpy as np
import glob


def rebm(config):
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

    total_x = x
    total_y = y
    # config["sgld_lr"] *= np.sqrt(np.prod(x.shape[1:]))

    # make a neural network to predict scores
    forward_model = ForwardModel(task.input_shape, hidden=config["hidden_size"])

    trainer = MaximumLikelihood(
        forward_model=forward_model,
        forward_model_opt=tf.keras.optimizers.Adam,
        forward_model_lr=config["forward_model_lr"],
        sgld_lr=config["sgld_lr"],
        sgld_noise_penalty=config["sgld_noise_penalty"],
        sgld_train_steps=config["sgld_train_steps"],
        sgld_eval_steps=config["sgld_eval_steps"],
        init_noise_std=config["init_noise_std"],
        is_discrete=config["is_discrete"],
        continuous_noise_std=config.get("continuous_noise_std", 0.0),
        discrete_smoothing=config.get("discrete_smoothing", 0.6),
    )

    # create a data set
    train_data, validate_data = task.build(
        x=x, y=y, batch_size=config["batch_size"], val_size=config["val_size"]
    )

    for epoch in range(config["epochs"]):
        statistics = defaultdict(list)
        for x, y in train_data:
            for name, tsr in trainer.train_step(x, y).items():
                statistics[name].append(tsr)

        for x, y in validate_data:
            for name, tsr in trainer.validate_step(x, y).items():
                statistics[name].append(tsr)

        for name in statistics.keys():
            logger.record(name, tf.concat(statistics[name], axis=0), epoch)