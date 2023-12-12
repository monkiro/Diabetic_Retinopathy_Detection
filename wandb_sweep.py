import logging
import wandb
import gin
import math

from input_pipeline.dataset import load
from model.vgg_like import *
from train import Trainer
from utils import logger, save


def train_func():
    with wandb.init() as run:
        gin.clear_config()
        # Hyperparameters
        bindings = []
        for key, value in run.config.items():
            bindings.append(f'{key}={value}')

        # # generate folder structures
        #
        # # dense_units_truncated = str(run.config.vgg_like.dense_units)[:3]
        # #dropout_rate_truncated = str(run.config['vgg_like.dropout_rate'])[:5]
        #
        # # bindings.append(f'vgg_like.dense_units={dense_units_truncated}')
        # #bindings.append(f'vgg_like.dropout_rate={dropout_rate_truncated}')
        #
        # run_paths = save.gen_run_folder(','.join(bindings))
        #
        # # set loggers
        # logger.set_loggers(run_paths['path_logs_train'], logging.INFO)
        #
        # # gin-config
        # gin.parse_config_files_and_bindings(['D:\\DL_Lab_P1\\config.gin'], bindings)
        # save.save_config(run_paths['path_gin'], gin.config_str())

        run_paths = save.gen_run_folder()  # without given id,generate itself by datetime

        # set loggers
        logger.set_loggers(run_paths['path_logs_train'], logging.INFO)

        # gin-config
        gin.parse_config_files_and_bindings(['D:\\DL_Lab_P1\\config.gin'], [])
        save.save_config(run_paths['path_gin'], gin.config_str())

        # setup pipeline
        ds_train, ds_val, ds_test, ds_info = load()

        # model
        model = vgg_like()

        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths)
        for _ in trainer.train():
            continue


sweep_config = {
    'name': 'vgg_example_sweep01',
    'method': 'random',
    'metric': {
        'name': 'val_acc',
        'goal': 'maximize'
    },

    'parameters': {
        'Trainer.total_steps': {
            'values': [800]
        },
        'vgg_like.base_filters': {
            'distribution': 'q_log_uniform',
            'q': 1,
            'min': math.log(8),
            'max': math.log(128)
        },
        'vgg_like.n_blocks': {
            'distribution': 'q_uniform',
            'q': 1,
            'min': 2,
            'max': 6
        },
        'vgg_like.dense_units': {
            'distribution': 'q_log_uniform',
            'q': 1,
            'min': math.log(16),
            'max': math.log(256)
        },
        'vgg_like.dropout_rate': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 0.6
        }
    }
}
sweep_id = wandb.sweep(sweep_config)

wandb.agent(sweep_id, function=train_func, count=10)



