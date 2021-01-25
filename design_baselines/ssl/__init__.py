from design_baselines.data import StaticGraphTask
from design_baselines.logger import Logger
from design_baselines.utils import spearman
from design_baselines.utils import soft_noise
from design_baselines.ssl.trainers import SemiSupervisedMaximumLikelihood
from design_baselines.ssl.nets import ForwardModel
from collections import defaultdict
import tensorflow as tf
import numpy as np
import glob


def ssl(config):
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

    # create the training task and logger
    train_data, val_data = task.build(
        x=x,
        y=y,
        bootstraps=config["bootstraps"],
        batch_size=config["batch_size"],
        val_size=config["val_size"],
    )

    # make several keras neural networks with two hidden layers
    forward_models = [
        ForwardModel(
            task.input_shape,
            hidden=config["hidden_size"],
            initial_max_std=config["initial_max_std"],
            initial_min_std=config["initial_min_std"],
        )
        for b in range(config["bootstraps"])
    ]

    # create a trainer for a forward model with a conservative objective
    trainer = SemiSupervisedMaximumLikelihood(
        forward_models,
        forward_model_optim=tf.keras.optimizers.Adam,
        forward_model_lr=config["lr"],
        augment=config["augment"],
        augment_std_penalty=config["augment_std_penalty"],
    )

    # select the top k initial designs from the dataset
    mean_x = tf.reduce_mean(x, axis=0, keepdims=True)
    indices = tf.math.top_k(y[:, 0], k=config["batch_size"])[1]
    initial_x = tf.gather(x, indices, axis=0)
    initial_x = (
        tf.math.log(soft_noise(initial_x, config.get("discrete_smoothing", 0.6)))
        if config["is_discrete"]
        else initial_x
    )

    trainer.solution = tf.Variable(initial_x)
    for epoch in range(config["epochs"]):
        trainer.launch(train_data, val_data, logger, 1, start_epoch=epoch)
        score = task.score(trainer.solution * st_x + mu_x)
        logger.record("score", score, epoch, percentile=True)


    """
    solver_lr = config["solver_lr"] * np.sqrt(np.prod(x.shape[1:]))
    solver_interval = int(
        config["solver_interval"] * (x.shape[0] - config["val_size"]) / config["batch_size"]
    )
    solver_warmup = int(
        config["solver_warmup"] * (x.shape[0] - config["val_size"]) / config["batch_size"]
    )

    # create a trainer for a forward model with a conservative objective
    trainer = EnergyBasedMaximumLikelihood(forward_model)

    # create a data set
    train_data, validate_data = task.build(
        x=x, y=y, batch_size=config["batch_size"], val_size=config["val_size"]
    )

    # select the top k initial designs from the dataset
    mean_x = tf.reduce_mean(x, axis=0, keepdims=True)
    indices = tf.math.top_k(y[:, 0], k=config["batch_size"])[1]
    initial_x = tf.gather(x, indices, axis=0)
    initial_x = (
        tf.math.log(soft_noise(initial_x, config.get("discrete_smoothing", 0.6)))
        if config["is_discrete"]
        else initial_x
    )

    # create the starting point for the optimizer
    evaluations = 0
    trainer.solution = tf.Variable(initial_x)
    trainer.w_normalizer = np.mean(np.exp(1.0 * y))

    # keep track of when to record performance
    interval = trainer.solver_interval
    warmup = trainer.solver_warmup

    # train model for many epochs
    for e in range(config["epochs"]):

        statistics = defaultdict(list)
        for x, y in train_data:
            for name, tensor in trainer.train_step(x, y).items():
                statistics[name].append(tensor)

        # evaluate the sampled designs
        score = task.score(trainer.solution * st_x + mu_x)
        logger.record("score", score, e, percentile=True)

        for name in statistics.keys():
            logger.record(name, tf.concat(statistics[name], axis=0), e)
    """