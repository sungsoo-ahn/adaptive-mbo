from design_baselines.data import StaticGraphTask
from design_baselines.logger import Logger
from design_baselines.utils import spearman
from design_baselines.utils import soft_noise
from design_baselines.utils import render_video
from design_baselines.sgld.trainers import MaximumLikelihood
from design_baselines.sgld.trainers import Ensemble
from design_baselines.sgld.nets import ForwardModel
from collections import defaultdict
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
import os
import glob


def sgld(config):
    """Train a Score Function to solve a Model-Based Optimization
    using gradient ascent on the input design

    Args:

    config: dict
        a dictionary of hyper parameters such as the learning rate
    """

    # create the training task and logger
    logger = Logger(config["logging_dir"])
    task = StaticGraphTask(config["task"], **config["task_kwargs"])

    # make several keras neural networks with different architectures
    forward_models = [
        ForwardModel(
            task.input_shape,
            activations=activations,
            hidden=config["hidden_size"],
            initial_max_std=config["initial_max_std"],
            initial_min_std=config["initial_min_std"],
        )
        for activations in config["activations"]
    ]

    # save the initial dataset statistics for safe keeping
    x = task.x
    y = task.y

    if config["normalize_ys"]:

        # compute normalization statistics for the score
        mu_y = np.mean(y, axis=0, keepdims=True)
        mu_y = mu_y.astype(np.float32)
        y = y - mu_y
        st_y = np.std(y, axis=0, keepdims=True)
        st_y = np.where(np.equal(st_y, 0), 1, st_y)
        st_y = st_y.astype(np.float32)
        y = y / st_y

    else:

        # compute normalization statistics for the score
        mu_y = np.zeros_like(y[:1])
        st_y = np.ones_like(y[:1])

    if config["normalize_xs"] and not config["is_discrete"]:

        # compute normalization statistics for the data vectors
        mu_x = np.mean(x, axis=0, keepdims=True)
        mu_x = mu_x.astype(np.float32)
        x = x - mu_x
        st_x = np.std(x, axis=0, keepdims=True)
        st_x = np.where(np.equal(st_x, 0), 1, st_x)
        st_x = st_x.astype(np.float32)
        x = x / st_x

    else:

        # compute normalization statistics for the score
        mu_x = np.zeros_like(x[:1])
        st_x = np.ones_like(x[:1])

    # scale the learning rate based on the number of channels in x
    config["solver_lr"] *= np.sqrt(np.prod(x.shape[1:]))
    config["solver_noise_rate"] *= np.sqrt(np.prod(x.shape[1:]))


    trs = []
    for i, fm in enumerate(forward_models):

        # create a bootstrapped data set
        train_data, validate_data = task.build(
            x=x, y=y, batch_size=config["batch_size"], val_size=config["val_size"], bootstraps=1
        )

        # create a trainer for a forward model with a conservative objective
        trainer = MaximumLikelihood(
            fm,
            forward_model_optim=tf.keras.optimizers.Adam,
            forward_model_lr=config["forward_model_lr"],
            is_discrete=config["is_discrete"],
            continuous_noise_std=config.get("continuous_noise_std", 0.0),
            discrete_smoothing=config.get("discrete_smoothing", 0.6),
        )

        # train the model for an additional number of epochs
        trs.append(trainer)
        trainer.launch(train_data, validate_data, logger, config["epochs"], header=f"oracle_{i}/")

    # select the top k initial designs from the dataset
    mean_x = tf.reduce_mean(x, axis=0, keepdims=True)
    indices = tf.math.top_k(y[:, 0], k=config["solver_samples"])[1]
    initial_x = tf.gather(x, indices, axis=0)
    x = (
        tf.math.log(soft_noise(initial_x, config.get("discrete_smoothing", 0.6)))
        if config["is_discrete"]
        else initial_x
    )

    # evaluate the starting point
    solution = tf.math.softmax(x) if config["is_discrete"] else x
    score = task.score(solution * st_x + mu_x)
    preds = [fm.get_distribution(solution).mean() * st_y + mu_y for fm in forward_models]

    # record the prediction and score to the logger
    logger.record("score", score, 0, percentile=True)
    logger.record("distance/travelled", tf.linalg.norm(solution - initial_x), 0)
    logger.record("distance/from_mean", tf.linalg.norm(solution - mean_x), 0)
    for n, prediction_i in enumerate(preds):
        logger.record(f"oracle_{n}/prediction", prediction_i, 0)
        logger.record(f"rank_corr/{n}_to_real", spearman(prediction_i[:, 0], score[:, 0]), 0)
        if n > 0:
            logger.record(f"rank_corr/0_to_{n}", spearman(preds[0][:, 0], prediction_i[:, 0]), 0)

    # perform gradient ascent on the score through the forward model
    for i in range(1, config["solver_steps"] + 1):
        # back propagate through the forward model
        with tf.GradientTape() as tape:
            tape.watch(x)
            predictions = []
            for fm in forward_models:
                solution = tf.math.softmax(x) if config["is_discrete"] else x
                predictions.append(fm.get_distribution(solution).mean())
            if config["aggregation_method"] == "mean":
                score = tf.reduce_mean(predictions, axis=0)
            if config["aggregation_method"] == "min":
                score = tf.reduce_min(predictions, axis=0)
            if config["aggregation_method"] == "random":
                score = predictions[np.random.randint(len(predictions))]

        grads = tape.gradient(score, x)

        # use the conservative optimizer to update the solution
        noise = tf.random.normal(x.shape, 0, 1)
        x = x + config["solver_lr"] * grads + config["solver_noise_rate"] * noise
        solution = tf.math.softmax(x) if config["is_discrete"] else x

        # evaluate the design using the oracle and the forward model
        score = task.score(solution * st_x + mu_x)
        preds = [fm.get_distribution(solution).mean() * st_y + mu_y for fm in forward_models]

        # record the prediction and score to the logger
        logger.record("score", score, i, percentile=True)
        for n, prediction_i in enumerate(preds):
            logger.record(f"oracle_{n}/prediction", prediction_i, i)
            logger.record(
                f"oracle_{n}/grad_norm",
                tf.linalg.norm(tf.reshape(grads[n], [-1, task.input_size]), axis=-1),
                i,
            )
            logger.record(f"rank_corr/{n}_to_real", spearman(prediction_i[:, 0], score[:, 0]), i)

        # save the best design to the disk
        np.save(os.path.join(config["logging_dir"], f"score_{i}.npy"), score)
        np.save(os.path.join(config["logging_dir"], f"solution_{i}.npy"), solution)