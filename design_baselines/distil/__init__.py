from design_baselines.data import StaticGraphTask
from design_baselines.logger import Logger
from design_baselines.utils import spearman
from design_baselines.utils import soft_noise, cont_noise
from design_baselines.distil.trainers import Trainer
from design_baselines.distil.nets import ForwardModel
from collections import defaultdict
import tensorflow as tf
import numpy as np
from tensorflow_probability import distributions as tfpd


def normalize_dataset(x, y, normalize_xs, normalize_ys):
    if normalize_ys:
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

    if normalize_xs:
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

    return (x, mu_x, st_x), (y, mu_y, st_y)


def distil(config):
    # create the training task and logger
    logger = Logger(config["logging_dir"])
    task = StaticGraphTask(config["task"], **config["task_kwargs"])
    if config["is_discrete"]:
        task_x = soft_noise(task.x, config["discrete_smoothing"])
        task_x = tf.math.log(task_x)
        task_y = task.y
    else:
        task_x = task.x
        task_y = task.y


    (task_x, mu_x, st_x), (task_y, mu_y, st_y) = normalize_dataset(
        x=task_x,
        y=task_y,
        normalize_xs=config["normalize_xs"],
        normalize_ys=config["normalize_ys"],
    )

    indices = tf.math.top_k(task_y[:, 0], k=config["sol_x_samples"])[1]
    init_sol_x =  tf.gather(task_x, indices, axis=0)
    sol_x_opt = tf.keras.optimizers.Adam(learning_rate=config["sol_x_lr"])

    pess_model = ForwardModel(
        input_shape=task.input_shape,
        hidden=config["hidden_size"],
        smoothing_rate=config["smoothing_rate"],
    )
    pess_model_opt = tf.keras.optimizers.Adam(learning_rate=config["model_lr"])
    distil_model = ForwardModel(
        input_shape=task.input_shape,
        hidden=config["hidden_size"],
        smoothing_rate=0.0, #config["smoothing_rate"],
    )
    distil_model_opt = tf.keras.optimizers.Adam(learning_rate=config["model_lr"])

    trainer = Trainer(
        pess_model=pess_model,
        pess_model_opt=pess_model_opt,
        distil_model=distil_model,
        distil_model_opt=distil_model_opt,
        init_sol_x=init_sol_x,
        sol_x_opt=sol_x_opt,
        coef_pessimism=config["coef_pessimism"],
        mc_evals=config["mc_evals"],
        )

    ### Warmup
    train_data, validate_data = task.build(
        x=task_x, y=task_y, batch_size=config["batch_size"], val_size=config["val_size"]
    )

    for epoch in range(config["warmup_epochs"]):
        statistics = defaultdict(list)
        for x, y in train_data:
            for name, tsr in trainer.train_pess_step(x, y).items():
                statistics[f"warmup/train/{name}"].append(tsr)

        for name, tsrs in statistics.items():
            logger.record(name, tf.reduce_mean(tf.concat(tsrs, axis=0)), epoch)

    distil_model.set_weights(pess_model.get_weights())

    distil_x = task_x
    distil_y = task_y
    pess_epoch = 0
    distil_epoch = 0
    for update in range(config["updates"]):
        # Update
        statistics = trainer.update_step()
        for name, tsr in statistics.items():
            logger.record(f"update/{name}", tsr, update)

        if (update + 1) % config["score_freq"] == 0:
            inp = tf.math.softmax(sol_x) if config["is_discrete"] else sol_x
            score = task.score(inp * st_x + mu_x)
            logger.record(f"update/score", score, update+1, percentile=True)

        # Train Pess
        for _ in range(config["pess_epochs_per_update"]):
            pess_epoch += 1
            statistics = defaultdict(list)
            for x, y in train_data:
                for name, tsr in trainer.train_pess_step(x, y).items():
                    statistics[f"pess/{name}"].append(tsr)

            for name, tsrs in statistics.items():
                logger.record(name, tf.reduce_mean(tf.concat(tsrs, axis=0)), pess_epoch)

        # Prepare Distil data
        statistics = trainer.inference_step()
        for name, tsr in statistics.items():
            logger.record(f"inference/{name}", tsr, update)

        sol_x = trainer.sol_x.read_value().numpy()
        sol_y = trainer.sol_y.read_value().numpy()
        distil_x = np.concatenate([sol_x, distil_x], axis=0)[:config["buffer_size"]]
        distil_y = np.concatenate([sol_y, distil_y], axis=0)[:config["buffer_size"]]
        distil_data, _ = task.build(
            x=distil_x, y=distil_y, batch_size=config["batch_size"], val_size=0
            )

        #Train Distil
        for epoch in range(config["distil_epochs_per_update"]):
            distil_epoch += 1
            statistics = defaultdict(list)
            for x, y in distil_data:
                for name, tsr in trainer.train_distil_step(x, y).items():
                    statistics[f"distil/{name}"].append(tsr)

            for name, tsrs in statistics.items():
                logger.record(name, tf.reduce_mean(tf.concat(tsrs, axis=0)), distil_epoch)

