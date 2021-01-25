from design_baselines.data import StaticGraphTask
from design_baselines.logger import Logger
from design_baselines.utils import spearman
from design_baselines.utils import soft_noise, cont_noise
from design_baselines.energy.trainers import MaximumLikelihood, EnergyMaximumLikelihood, Buffer
from design_baselines.energy.nets import ForwardModel
from collections import defaultdict
import tensorflow as tf
import numpy as np
import glob


def energy(config):
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
    forward_model = ForwardModel(
        task.input_shape, activations=config["activations"], hidden=config["hidden_size"],
    )
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
        sgld_lr=config["sgld_lr"],
        sgld_noise_penalty=config["sgld_noise_penalty"],
        alpha=config["alpha"],
        is_discrete=config["is_discrete"],
        continuous_noise_std=config.get("continuous_noise_std", 0.0),
        discrete_smoothing=config.get("discrete_smoothing", 0.6),
    )

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

    def get_topk_samples(num_samples):
        indices = tf.math.top_k(total_y[:, 0], k=config["batch_size"])[1]
        topk_samples = tf.gather(total_x, indices, axis=0)
        topk_samples = (
            soft_noise(topk_samples, config["discrete_smoothing"])
            if config["is_discrete"]
            else cont_noise(topk_samples, config["continuous_noise_std"])
        )
        return topk_samples

    def run_markovchains(x, num_steps, log_freq=0, prefix=""):
        for step in range(num_steps):
            x, energy = trainer.run_markovchain(x)
            if log_freq > 0 and (step + 1) % log_freq == 0:
                score = task.score(st_x * x[: config["batch_size"]] + mu_x)
                logger.record(
                    f"{prefix}/energy", energy[: config["batch_size"]], step, percentile=True
                )
                logger.record(
                    f"{prefix}/pred_score",
                    st_y * energy[: config["batch_size"]] + mu_y,
                    step,
                    percentile=True,
                )
                logger.record(
                    f"{prefix}/score", score[: config["batch_size"]], step, percentile=True
                )

        return x

    # create a data set
    train_data, validate_data = task.build(
        x=x, y=y, batch_size=config["batch_size"], val_size=config["val_size"]
    )

    # Pretrain
    step = 0
    for e in range(config["pretrain_epochs"]):
        for x, y in train_data:
            statistics = pretrainer.train_step(x, y)
            step += 1
            for name in statistics.keys():
                logger.record(name, tf.concat(statistics[name], axis=0), step)

    # Initialize buffer with warmup samples
    bufferwarmup_cnt = 0
    while not buffer.full():
        num_samples = min(config["buffer_update_size"], buffer.buffer_size - len(buffer))
        x = get_init_samples(num_samples=num_samples)
        x = run_markovchains(
            x,
            num_steps=config["warmup_steps"],
            log_freq=config["log_freq"] if buffer.empty() else 0,
            prefix=f"buffer_warmup",
        )
        buffer.add(x)
        logger.record(f"buffer_warmup/buffer_size", len(buffer), bufferwarmup_cnt)
        bufferwarmup_cnt += 1

    # Check initial performance
    x = get_topk_samples(num_samples=config["batch_size"])
    x = run_markovchains(
        x, num_steps=config["warmup_steps"], log_freq=config["log_freq"], prefix="start"
    )
    score = task.score(st_x * x + mu_x)
    logger.record(f"start/score", score, 0, percentile=True)

    # Train
    step = 0
    for epoch in range(config["epochs"]):
        for batch_idx, (x, y) in enumerate(train_data):
            x_neg = buffer.sample(num_samples=config["batch_size"])
            x_neg = run_markovchains(x_neg, num_steps=config["pcd_steps"])
            buffer.add(x_neg)
            statistics = trainer.train_step(x, y, x_neg)
            step += 1
            for name in statistics.keys():
                logger.record(name, tf.concat(statistics[name], axis=0), step)

            if config["log_freq"] > 0 and (step + 1) % config["log_freq"] == 0:
                score = task.score(st_x * x_neg + mu_x)
                logger.record(f"train/score", score, step, percentile=True)

            if (step + 1) % config["buffer_update_freq"] == 0:
                tag = step + 1 // config["buffer_update_freq"]
                x = get_init_samples(num_samples=config["buffer_update_size"])
                x = run_markovchains(
                    x,
                    num_steps=config["warmup_steps"],
                    log_freq=config["log_freq"],
                    prefix=f"buffer_update_{tag}",
                )
                buffer.remove(num_samples=config["buffer_update_size"])
                buffer.add(x)

    x_list = []
    energy_list = []
    while not buffer.empty():
        x = buffer.sample(num_samples=config["batch_size"])
        energy = forward_model(tf.math.softmax(x) if config["is_discrete"] else x)
        x_list.append(x)
        energy_list.append(energy)

    """
    x = tf.concat(x_list, axis=0)
    energy = tf.concat(energy_list, axis=0)
    indices = tf.math.top_k(energy[:, 0], k=config["batch_size"])[1]
    topk_x = tf.gather(x, indices, axis=0)
    solution = tf.math.softmax(topk_x) if config["is_discrete"] else topk_x
    solution = st_x * solution + mu_x
    score = task.score(solution)
    """

    logger.record("final/buffer_score", score, 0, percentile=True)

    #Check final performance
    x = get_topk_samples(num_samples=config["batch_size"])
    x = run_markovchains(
        x, num_steps=config["warmup_steps"], log_freq=config["log_freq"], prefix="final"
    )
    score = task.score(st_x * x + mu_x)
    logger.record(f"final/sampled_score", score, 0, percentile=True)

    """
    trainer.sgld_noise_penalty *= 0.001
    x = get_topk_samples(num_samples=config["batch_size"])
    x = run_markovchains(
        x, num_steps=config["warmup_steps"], log_freq=config["log_freq"], prefix="final"
    )
    score = task.score(st_x * x + mu_x)
    logger.record(f"final/maximized_score", score, 0, percentile=True)
    """

    # logger.record("eval/score", score, e, percentile=True)
    # logger.record("eval/score", score, e, percentile=False)
    # energy_pos = forward_model(tf.math.softmax(solution) if config["is_discrete"] else solution)
    # y_pred = energy_pos / config["energy_scale"]
    # logger.record("eval/score_pred", y_pred * st_y + mu_y, e, percentile=True)

