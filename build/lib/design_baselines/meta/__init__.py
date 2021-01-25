from design_baselines.data import StaticGraphTask
from design_baselines.logger import Logger
from design_baselines.utils import spearman
from design_baselines.utils import soft_noise, cont_noise
from design_baselines.utils import render_video
from design_baselines.meta.trainers import MaximumLikelihood
from design_baselines.meta.nets import UnrolledForwardModel
from collections import defaultdict
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
import os
import glob


def meta(config):
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

    total_x = x
    total_y = y

    # create a bootstrapped data set
    train_data, validate_data = task.build(
        x=total_x, y=total_y, batch_size=config["batch_size"], val_size=config["val_size"],
    )

    #model = ForwardModel(**model_config)
    model = UnrolledForwardModel(
        input_shape=task.input_shape,
        hidden=config["hidden_size"],
        anchor_coef=config["anchor_coef"],
        )

    # create a trainer for a forward model with a conservative objective
    trainer = MaximumLikelihood(
        model,
        model_optim=tf.keras.optimizers.Adam,
        model_lr=config["model_lr"],
        unrolls=config["unrolls"],
        unroll_rate=config["unroll_rate"],
        unrolled_coef=config["unrolled_coef"],
        is_discrete=config["is_discrete"],
        continuous_noise_std=config.get("continuous_noise_std", 0.0),
        discrete_smoothing=config.get("discrete_smoothing", 0.6),
    )

    for epoch in range(config["epochs"]):
        statistics = defaultdict(list)
        #trainer.unrolls = int(np.ceil(config["unrolls"] * epoch / config["epochs"]))
        #statistics["unrolls"] = trainer.unrolls
        for x, y in train_data:
            if epoch < config["warmup"]:
                for name, tsr in trainer.warmup_train_step(x, y).items():
                    statistics[name].append(tsr)
            else:
                for name, tsr in trainer.train_step(x, y).items():
                    statistics[name].append(tsr)

        for x, y in validate_data:
            if epoch < config["warmup"]:
                for name, tsr in trainer.warmup_validate_step(x, y).items():
                    statistics[name].append(tsr)
            else:
                for name, tsr in trainer.validate_step(x, y).items():
                    statistics[name].append(tsr)

        for name, tsrs in statistics.items():
            logger.record(name, tf.reduce_mean(tf.concat(tsrs, axis=0)), epoch)

    indices = tf.math.top_k(total_y[:, 0], k=config["solver_samples"])[1]
    initial_x = tf.gather(total_x, indices, axis=0)
    initial_x = (
        tf.math.log(soft_noise(initial_x, config.get("discrete_smoothing", 0.6)))
        if config["is_discrete"]
        else initial_x
    )
    x = tf.Variable(initial_x)
    if config["update_optim"] == "adam":
        x_optim = tf.keras.optimizers.Adam(learning_rate=config["update_rate"])
    elif config["update_optim"] == "sgd":
        x_optim = tf.keras.optimizers.SGD(learning_rate=config["update_rate"])

    @tf.function(experimental_relax_shapes=True)
    def update_x():
        with tf.GradientTape() as tape:
            tape.watch(x)
            inp = tf.math.softmax(x) if config["is_discrete"] else x
            unrolled_ds = model.get_unrolled_distributions(
                inp, unrolls=config["unrolls"], unroll_rate=config["unroll_rate"]
                )
            x_loss = - unrolled_ds[-1].mean()

        x_grad = tape.gradient(x_loss, x)
        x_optim.apply_gradients([[x_grad, x]])

        y_pred = unrolled_ds[0].mean()
        unrolled_y_pred = unrolled_ds[-1].mean()

        score_pred = y_pred * st_y + mu_y
        unrolled_score_pred = unrolled_y_pred * st_y + mu_y

        travelled = tf.linalg.norm(x - initial_x)

        statistics = dict()

        statistics["update/y_pred"] = tf.reduce_mean(y_pred)
        statistics["update/unrolled_y_pred"] = tf.reduce_mean(unrolled_y_pred)
        statistics["update/unrolled_delta"] =  tf.reduce_mean(y_pred - unrolled_y_pred)

        statistics["update/score_pred"] =  score_pred
        statistics["update/unrolled_score_pred"] =  unrolled_score_pred

        statistics["update/travelled"] =  travelled

        return statistics

    for update in range(config["updates"]):
        statistics = update_x()
        for name, tsr in statistics.items():
            logger.record(name, tsr, update)

        if (update + 1) % config["score_freq"] == 0:
            inp = tf.math.softmax(x) if config["is_discrete"] else x
            score = task.score(inp * st_x + mu_x)
            logger.record("update/score", score, update, percentile=True)


    """
    # select the top k initial designs from the dataset


    for epoch in range(config["epoch"]):
        for (x, y, _) in train_data:
            inp = tf.math.softmax(x) if config["is_discrete"] else x
            with tf.GradientTape as tape:
                meta_model = model.get_unrolled_model(
                    inp, steps=config["meta_steps"], step_size=config["meta_step_size"]
                    )
                d = meta_model.get_distribution(x)
                nll = d.log_prob(y)

            grads = tape.gradient(model.trainable_variables)
            model_opt.apply_gradients(zip(grads, model.trainable_variables))
    """

    """
    for update in range(config["update"]):
        with tf.GradientTape() as tape:
            tape.watch(x)
            inp = tf.math.softmax(x) if config["is_discrete"] else x
            unrolled_pred, meta_model, meta_statistics = model.get_unrolled_model(
                inp,
                steps=config["meta_steps"],
                step_size=config["meta_step_size"],
            )
            x_loss = -unrolled_pred

        x_grad = tape.gradient(x_loss, x)
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