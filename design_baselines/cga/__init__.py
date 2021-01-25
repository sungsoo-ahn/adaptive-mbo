from design_baselines.data import StaticGraphTask
from design_baselines.logger import Logger
from design_baselines.utils import spearman
from design_baselines.utils import soft_noise
from design_baselines.cga.trainers import ConservativeMaximumLikelihood
from design_baselines.cga.nets import ForwardModel
from collections import defaultdict
import tensorflow as tf
import numpy as np
import glob


def cga(config):
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

    #solver_lr = config["solver_lr"] * np.sqrt(np.prod(x.shape[1:]))

    # make a neural network to predict scores
    forward_model = ForwardModel(
        task.input_shape,
        activations=config["activations"],
        hidden=config["hidden_size"],
        initial_max_std=config["initial_max_std"],
        initial_min_std=config["initial_min_std"],
    )

    # create a trainer for a forward model with a conservative objective
    trainer = ConservativeMaximumLikelihood(
        forward_model,
        forward_model_opt=tf.keras.optimizers.Adam,
        forward_model_lr=config["forward_model_lr"],
        alpha=config["alpha"],
        solver_lr=config["solver_lr"],
        solver_noise_penalty=config["solver_noise_penalty"],
        is_discrete=config["is_discrete"],
        continuous_noise_std=config.get("continuous_noise_std", 0.0),
        discrete_smoothing=config.get("discrete_smoothing", 0.6),
    )

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
    trainer.solution = tf.Variable(initial_x)

    #
    def evaluate_solution(solution):
        statistics = dict()
        solution = tf.math.softmax(solution) if config["is_discrete"] else solution
        score = task.score(solution * st_x + mu_x)
        statistics["eval/score"] = score
        return statistics

    # train model for many epochs with conservatism
    for epoch in range(config["epochs"]):
        statistics = defaultdict(list)
        warmup = (epoch + 1) > config["warmup"]
        for x, y in train_data:
            for name, tensor in trainer.train_step(x, y).items():
                statistics[name].append(tensor)

        if warmup and (epoch + 1) % config["solver_update_freq"] == 0:
            for name, tensor in trainer.update_solution().items():
                statistics[name].append(tensor)

        if warmup and (epoch + 1) % config["eval_freq"] == 0:
            for name, tensor in evaluate_solution(trainer.solution).items():
                statistics[name].append(tensor)

        for name in statistics.keys():
            if name in ["eval/score"]:
                logger.record(name, tf.concat(statistics[name], axis=0), epoch, percentile=True)
            else:
                logger.record(name, tf.concat(statistics[name], axis=0), epoch, percentile=False)
