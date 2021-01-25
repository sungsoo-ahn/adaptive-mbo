from design_baselines.data import StaticGraphTask
from design_baselines.logger import Logger
from design_baselines.utils import spearman
from design_baselines.utils import soft_noise, cont_noise
from design_baselines.utils import render_video
from design_baselines.clip.trainers import MaximumLikelihood
from design_baselines.clip.nets import ForwardModel
from collections import defaultdict
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
import os
import glob


def clip(config):
    """Train a Score Function to solve a Model-Based Optimization
    using gradient ascent on the input design

    Args:

    config: dict
        a dictionary of hyper parameters such as the learning rate
    """

    # create the training task and logger
    logger = Logger(config["logging_dir"])
    task = StaticGraphTask(config["task"], **config["task_kwargs"])

    # save the initial dataset statistics for safe keeping
    x = task.x
    y = task.y

    #config["step_size"] *= np.sqrt(np.prod(x.shape[1:]))

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
    config["step_size"] *= np.sqrt(np.prod(x.shape[1:]))

    # make several keras neural networks with different architectures
    model_config = {
        "input_shape": task.input_shape,
        # "activations": config["activations"],
        "hidden": config["hidden_size"],
        "spectral_normalization": config["spectral_normalization"],
    }

    # create a bootstrapped data set
    train_data, validate_data = task.build(
        x=x, y=y, batch_size=config["batch_size"], val_size=config["val_size"], bootstraps=1,
    )

    model = ForwardModel(**model_config)

    # create a trainer for a forward model with a conservative objective
    trainer = MaximumLikelihood(
        model,
        forward_model_optim=tf.keras.optimizers.Adam,
        forward_model_lr=config["forward_model_lr"],
        bootstrap_id=0,
        is_discrete=config["is_discrete"],
        continuous_noise_std=config.get("continuous_noise_std", 0.0),
        discrete_smoothing=config.get("discrete_smoothing", 0.6),
    )

    # train the model for an additional number of epochs
    trainer.launch(
        train_data, validate_data, logger, config["epochs"], header=f"pretrain_model/",
    )
    init_weights = model.get_weights()

    """
    lambdas = [config["min_lambda"] for _ in model.get_meta_weights()]
    for sample_idx, (xi, yi, _) in enumerate(train_data):
        #xi = np.expand_dims(xi, 0)
        #yi = np.expand_dims(yi, 0)
        xi0 = (
            soft_noise(xi, trainer.keep)
            if config["is_discrete"]
            else cont_noise(xi, trainer.noise_std)
        )
        with tf.GradientTape() as tape:
            yi_pred = model(xi0)
            mse = tf.reduce_mean(tf.square(yi-yi_pred))

        grads = tape.gradient(mse, model.get_meta_weights())
        square_grads = [tf.square(grad) for grad in grads]
        lambdas = [lambda_ + square_grad for lambda_, square_grad in zip(lambdas, square_grads)]

    for lambda_idx, lambda_ in enumerate(lambdas):
        logger.record(f"lambda/grad2_layer{lambda_idx}", tf.reduce_mean(lambda_), 0)

    flat_lambda = tf.concat([tf.reshape(lambda_, [-1]) for lambda_ in lambdas], axis=0)
    max_lambda, min_lambda = tf.reduce_max(flat_lambda), tf.reduce_min(flat_lambda)
    lambdas = [(lambda_ - min_lambda) / (max_lambda - min_lambda) for lambda_ in lambdas]
    lambdas = [
        config["min_lambda"] + lambda_ * (config["max_lambda"] - config["min_lambda"])
        for lambda_ in lambdas
        ]
    """

    lambdas = [config["lambda"] for _ in model.get_meta_weights()]

    # select the top k initial designs from the dataset
    indices = tf.math.top_k(y[:, 0], k=config["solver_samples"])[1]
    initial_x = tf.gather(x, indices, axis=0)
    x = (
        tf.math.log(soft_noise(initial_x, config.get("discrete_smoothing", 0.6)))
        if config["is_discrete"]
        else initial_x
    )
    x = tf.Variable(x)

    # evaluate the starting point
    # inp = tf.math.softmax(x) if config["is_discrete"] else x
    # score = task.score(inp * st_x + mu_x)
    # logger.record("score", score, 0, percentile=True)

    if config["optimizer"] == "adam":
        x_optim = tf.keras.optimizers.Adam(learning_rate=config["step_size"])
    elif config["optimizer"] == "sgd":
        x_optim = tf.keras.optimizers.SGD(learning_rate=config["step_size"])

    for step in range(config["steps"]):
        with tf.GradientTape() as tape:
            tape.watch(x)
            inp = tf.math.softmax(x) if config["is_discrete"] else x
            unrolled_pred, meta_model, meta_statistics = model.get_unrolled_pred(
                inp,
                lambdas,
                steps=config["meta_steps"],
                step_size=config["meta_step_size"],
                #unroll_type="sgd",
            )
            x_loss = -unrolled_pred

        x_grad = tape.gradient(x_loss, x)
        #x_grad = tf.clip_by_norm(x_grad, 1.0)
        x_optim.apply_gradients([[x_grad, x]])

        if (step + 1) % config["log_freq"] == 0:
            if config["meta_steps"] > 0:
                for name, tsr_list in meta_statistics.items():
                    for meta_step, tsr in enumerate(tsr_list):
                        logger.record(f"meta/{step:02d}/{name}", tsr, meta_step)

            inp = tf.math.softmax(x) if config["is_discrete"] else x
            score = task.score(inp * st_x + mu_x)
            y_pred = meta_model.get_distribution(inp).mean()
            score_pred = y_pred * st_y + mu_y
            travelled = tf.linalg.norm(inp - initial_x)
            logger.record("train/score_pred", score_pred, step, percentile=True)
            logger.record("train/score", score, step, percentile=True)
            logger.record("train/travelled", travelled, step)

            """
            trainer.fm = meta_model
            validate_statistics = trainer.validate(validate_data)
            for name, tsr in validate_statistics.items():
                logger.record(f"validate/{name}", tsr, step)
            """

