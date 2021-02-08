from design_baselines.data import StaticGraphTask
from design_baselines.logger import Logger
from design_baselines.utils import soft_noise, cont_noise
from design_baselines.ensemble.trainers import ModelTrainer, SolutionTrainer
from design_baselines.ensemble.nets import ForwardModel
from collections import defaultdict
import tensorflow_probability as tfp
import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
import os


def ensemble(config):
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

    total_x = x
    total_y = y

    # create a bootstrapped data set
    train_data, validate_data = task.build(
        x=x,
        y=y,
        batch_size=config["batch_size"],
        val_size=config["val_size"],
        bootstraps=config["ensemble_size"],
    )

    model_weights_list = []

    for model_id in range(config["ensemble_size"]):
        model = ForwardModel(input_shape=task.input_shape, hidden=config["hidden_size"])
        model_optim = tfa.optimizers.AdamW(weight_decay=config["model_wd"], learning_rate=config["model_lr"])
        #tf.keras.optimizers.Adam(learning_rate=config["model_lr"])
        trainer = ModelTrainer(
            model=model,
            model_optim=model_optim,
            is_discrete=config["is_discrete"],
            continuous_noise_std=config.get("continuous_noise_std", 0.0),
            discrete_smoothing=config.get("discrete_smoothing", 0.6),
        )

        for epoch in range(config["epochs"]):
            train_statistics = defaultdict(list)
            for x, y, b in train_data:
                for name, tsr in trainer.train_step(x, y, b).items():
                    train_statistics[name].append(tsr)

            for name, tsrs in train_statistics.items():
                logger.record(
                    f"train/model{model_id}/{name}", tf.reduce_mean(tf.concat(tsrs, axis=0)), epoch
                )

            validate_statistics = defaultdict(list)
            for x, y in validate_data:
                for name, tsr in trainer.validate_step(x, y).items():
                    validate_statistics[name].append(tsr)

            for name, tsrs in validate_statistics.items():
                logger.record(
                    f"validate/model{model_id}/{name}",
                    tf.reduce_mean(tf.concat(tsrs, axis=0)),
                    epoch,
                )

        model_weights = model.get_weights()
        model_weights_list.append(model_weights)

    indices = tf.math.top_k(total_y[:, 0], k=config["solver_samples"])[1]
    sol = tf.gather(total_x, indices, axis=0)
    sol = (
        tf.math.log(soft_noise(sol, config["discrete_smoothing"]))
        if config["is_discrete"]
        else sol
    )
    sol = tf.Variable(sol)
    sol_optim = tf.keras.optimizers.Adam(learning_rate=config["sol_lr"])

    sol_trainer = SolutionTrainer(
        model=model,
        model_weights_list=model_weights_list,
        sol=sol,
        sol_optim=sol_optim,
        is_discrete=config["is_discrete"],
        continuous_noise_std=config.get("continuous_noise_std", 0.0),
        discrete_smoothing=config.get("discrete_smoothing", 0.6),
    )

    for update in range(config["updates"]):
        statistics = sol_trainer.update_solution()
        if (update + 1) % config["score_freq"] == 0:
            inp = tf.math.softmax(sol_trainer.sol) if config["is_discrete"] else sol_trainer.sol
            score = task.score(inp * st_x + mu_x)
            y = (score - mu_y) / st_y
            statistics["score"] = score

        for name, tsrs in statistics.items():
            logger.record(f"update/{name}", tf.reduce_mean(tsrs), update)


