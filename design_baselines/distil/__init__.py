from design_baselines.data import StaticGraphTask
from design_baselines.logger import Logger
from design_baselines.utils import spearman
from design_baselines.utils import soft_noise, cont_noise
from design_baselines.distil.trainers import PessimisticTrainer, MainTrainer
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
    sol_x =  tf.gather(task_x, indices, axis=0)
    sol_x_opt = tf.keras.optimizers.Adam(learning_rate=config["sol_x_lr"])

    perturb_fn = lambda x: cont_noise(x, noise_std=config["continuous_noise_std"])
    main_model = ForwardModel(
        input_shape=task.input_shape,
        hidden=config["hidden_size"],
        is_discrete=config["is_discrete"],
    )
    main_model_opt = tf.keras.optimizers.Adam(learning_rate=config["model_lr"])
    main_trainer = MainTrainer(
        model=main_model,
        model_opt=main_model_opt,
        perturb_fn=perturb_fn,
        sol_x=sol_x,
        sol_x_opt=sol_x_opt,
        )

    pessimistic_model = ForwardModel(
        input_shape=task.input_shape,
        hidden=config["hidden_size"],
        is_discrete=config["is_discrete"],
    )
    pessimistic_model_opt = tf.keras.optimizers.Adam(learning_rate=config["model_lr"])
    pessimistic_trainer = PessimisticTrainer(
        model=pessimistic_model,
        model_opt=pessimistic_model_opt,
        perturb_fn=perturb_fn,
        sol_x=sol_x,
        coef_pessimism=config["coef_pessimism"],
        ema_rate=config["ema_rate"],
    )

    ### Warmup
    task_train_data, task_validate_data = task.build(
        x=task_x, y=task_y, batch_size=config["batch_size"], val_size=config["val_size"]
    )

    for epoch in range(config["warmup_epochs"]):
        statistics = defaultdict(list)
        for x, y in task_train_data:
            for name, tsr in main_trainer.train_step(x, y).items():
                statistics[f"warmup/train/{name}"].append(tsr)

        for x, y in task_validate_data:
            for name, tsr in main_trainer.validate_step(x, y).items():
                statistics[f"warmup/validate/{name}"].append(tsr)

        for name, tsrs in statistics.items():
            logger.record(name, tf.reduce_mean(tf.concat(tsrs, axis=0)), epoch)

    warmup_weights = main_model.get_weights()
    pessimistic_model.set_weights(warmup_weights)

    ### Main training
    total_x = task_x
    total_y = task_y
    pessimistic_epoch = 0
    main_epoch = 0
    for update in range(config["updates"]):
        pessimistic_trainer.assign_sol_x(sol_x)
        pessimistic_trainer.init_sol_y()
        for _ in range(config["pessimistic_epochs_per_update"]):
            pessimistic_epoch += 1
            statistics = defaultdict(list)
            for x, y in task_train_data:
                for name, tsr in pessimistic_trainer.train_step(x, y).items():
                    statistics[f"pessimistic/{name}"].append(tsr)

            for name, tsrs in statistics.items():
                logger.record(name, tf.reduce_mean(tf.concat(tsrs, axis=0)), pessimistic_epoch)

        sol_y = pessimistic_trainer.get_sol_y()
        logger.record(f"update/sol_y", sol_y, update, percentile=True)

        total_x = tf.concat([total_x, sol_x], axis=0).numpy()
        total_y = tf.concat([total_y, sol_y], axis=0).numpy()
        total_x = total_x[:1024]
        total_y = total_y[:1024]
        main_data, _ = task.build(x=total_x, y=total_y, batch_size=config["batch_size"], val_size=0)

        for epoch in range(config["main_epochs_per_update"]):
            main_epoch += 1
            statistics = defaultdict(list)
            for x, y in main_data:
                for name, tsr in main_trainer.train_step(x, y).items():
                    statistics[f"main/{name}"].append(tsr)

            for name, tsrs in statistics.items():
                logger.record(name, tf.reduce_mean(tf.concat(tsrs, axis=0)), main_epoch)

        statistics = main_trainer.update_step()
        for name, tsr in statistics.items():
            logger.record(f"update/{name}", tsr, update+1)

        sol_x = main_trainer.get_sol_x()
        if (update + 1) % config["score_freq"] == 0:
            inp = tf.math.softmax(sol_x) if config["is_discrete"] else sol_x
            score = task.score(inp * st_x + mu_x)
            logger.record(f"update/score", score, update+1, percentile=True)
