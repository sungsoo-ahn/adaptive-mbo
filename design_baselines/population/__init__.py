from design_baselines.data import StaticGraphTask
from design_baselines.logger import Logger
from design_baselines.utils import spearman
from design_baselines.utils import soft_noise, cont_noise
from design_baselines.population.trainers import MaximumLikelihood, EnergyMaximumLikelihood, Buffer
from design_baselines.population.nets import ForwardModel
from collections import defaultdict
import tensorflow as tf
import numpy as np
import glob


def population(config):
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

    # create a trainer for a forward model with a conservative objective
    def get_init_samples(num_samples):
        indices = tf.random.uniform(shape=[num_samples], maxval=total_x.shape[0], dtype=tf.int32)
        init_samples = tf.gather(total_x, indices, axis=0)
        init_samples = (
            soft_noise(init_samples, config["discrete_smoothing"])
            if config["is_discrete"]
            else cont_noise(init_samples, config["continuous_noise_std"])
        )
        return init_samples

    # make a neural network to predict scores
    forward_model = ForwardModel(task.input_shape, hidden=config["hidden_size"])
    pretrainer = MaximumLikelihood(
        forward_model=forward_model,
        forward_model_opt=tf.keras.optimizers.Adam,
        forward_model_lr=config["forward_model_lr"],
        is_discrete=config["is_discrete"],
        continuous_noise_std=config.get("continuous_noise_std", 0.0),
        discrete_smoothing=config.get("discrete_smoothing", 0.6),
    )

    buffer = Buffer(buffer_size=config["buffer_size"])
    trainer = EnergyMaximumLikelihood(
        forward_model=forward_model,
        forward_model_opt=tf.keras.optimizers.Adam,
        forward_model_lr=config["forward_model_lr"],
        x_lr=config["x_lr"],
        alpha=config["alpha"],
        is_discrete=config["is_discrete"],
        continuous_noise_std=config.get("continuous_noise_std", 0.0),
        discrete_smoothing=config.get("discrete_smoothing", 0.6),
    )

    # create a data set
    train_data, validate_data = task.build(
        x=total_x, y=total_y, batch_size=config["batch_size"], val_size=config["val_size"]
    )

    # Pretrain
    for epoch in range(config["pretrain_epochs"]):
        statistics = defaultdict(list)
        for x, y in train_data:
            for name, tsr in pretrainer.train_step(x, y).items():
                statistics[name].append(tsr)

        for name in statistics.keys():
            logger.record(name, tf.concat(statistics[name], axis=0), epoch)

    # Initialize buffer
    buffer.add(get_init_samples(num_samples=buffer.buffer_size))

    # Train
    for epoch in range(config["epochs"]):
        statistics = defaultdict(list)
        for batch_idx, (x, y) in enumerate(train_data):
            x_neg = buffer.sample(num_samples=config["batch_size"], remove=True)
            for _ in range(config["x_steps"]):
                x_neg, energy = trainer.run_markovchain(x_neg)

            buffer.add(x_neg)

            for name, tsr in trainer.train_step(x, y, x_neg).items():
                statistics[name].append(tsr)

        for name in statistics.keys():
            logger.record(
                name, tf.concat(statistics[name], axis=0), epoch
                )

        if (epoch + 1) % 500 == 0:
            solution = buffer.sample(num_samples=config["batch_size"], remove=False)
            solution = tf.math.softmax(solution) if config["is_discrete"] else solution
            solution = st_x * solution + mu_x
            score = task.score(solution)
            logger.record("score/batch", score, epoch, percentile=True)

            x_list = []
            energy_list = []
            while sum([x_.shape[0] for x_ in x_list]) < buffer.buffer_size:
                x = buffer.sample(num_samples=config["batch_size"], remove=False)
                energy = forward_model(tf.math.softmax(x) if config["is_discrete"] else x)
                x_list.append(x)
                energy_list.append(energy)

            x = tf.concat(x_list, axis=0)
            energy = tf.concat(energy_list, axis=0)
            indices = tf.math.top_k(energy[:, 0], k=config["batch_size"])[1]
            topk_x = tf.gather(x, indices, axis=0)
            solution = tf.math.softmax(topk_x) if config["is_discrete"] else topk_x
            solution = st_x * solution + mu_x
            score = task.score(solution)

            assert solution.shape[0] == config["batch_size"]
            logger.record("score/buffer", score, epoch, percentile=True)
            logger.record("score/buffer_noperc", score, epoch)
