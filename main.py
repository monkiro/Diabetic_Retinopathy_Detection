import os
# import sys
# sys.path.append("./diabetic_retinopathy")
import argparse
import random
import gin
import logging
import numpy as np
from absl import app, flags
import tensorflow as tf
# from Deep_visualization.Dimensionality_reduction import Dimensionality_Reduction
# from Input_pipeline.TFrecord_writer import write_Tfrecord
# from Input_pipeline.data_prepare import processing_augmentation_oversampling
from train import Trainer
from evaluation.evaluate_loss import evaluate, evaluate_fl
from input_pipeline.dataset import load, get_dataset
from utils import save, logger
from model.basic_CNN import *
from model.vgg_like import *
from model.vgg import *
from model.resnet import *
# from models.TL import tl_inception, tl_xception, tl_inception_resnet
# from evaluation.metrics import confusionmatrix, ROC

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--model', choices=['Basic_CNN','vgg_like', 'vgg', 'resnet', 'tl_inception', 'tl_xception', 'tl_inception_resnet'],
                    default='resnet', help='choose model')
parser.add_argument('--mode', choices=['train', 'test'], default='test', help='train or test')
parser.add_argument('--evaluation', choices=['evaluate_fl', 'confusionmatrix', 'Dimensionality_Reduction', 'ROC'],
                        default='evaluate_fl', help='evaluation methods')
parser.add_argument('--checkpoint_file', type=str, default='D:\\DL_Lab_P1\\ckpts\\resnet01\\',
                    help='Path to checkpoint.')

args = parser.parse_args()

print(args.model)
print(args.mode)
print(args.evaluation)
print(args.checkpoint_file)

# fix the seed to make the training repeatable
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  #主要是为了可重复性，实际上降低了效率


setup_seed(66)


@gin.configurable
def main(argv):
    # generate folder structures
    run_paths = save.gen_run_folder() #without given id,generate itself by datetime

    # set loggers
    logger.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['D:\\DL_Lab_P1\\config.gin'], [])
    save.save_config(run_paths['path_gin'], gin.config_str())


    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = load()

    if args.model == 'Basic_CNN':
        model = Basic_CNN()
    elif args.model == 'vgg_like':
        model = vgg_like()
    elif args.model == 'vgg':
        model_instance = VGG16Model()  # 创建类的实例
        model = model_instance.vgg()  # 调用实例方法
    elif args.model == 'resnet':
        model = resnet(input_shape=ds_info.input_shape, n_classes=ds_info.n_classes)
    # elif args.model == 'tl_inception':
    #     model = tl_inception(input_shape=ds_info.input_shape, n_classes=ds_info.n_classes)
    # elif args.model == 'tl_xception':
    #     model = tl_xception(input_shape=ds_info.input_shape, n_classes=ds_info.n_classes)
    # elif args.model == 'tl_inception_resnet':
    #     model = tl_inception_resnet(input_shape=ds_info.input_shape, n_classes=ds_info.n_classes)
    else:
        print('Error, model does not exist')

    model.summary()

    if args.mode == 'train':
        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths)

        for _ in trainer.train():
            continue
    else:
        checkpoint = tf.train.Checkpoint(step=tf.Variable(0), model=model)

        checkpoint.restore(os.path.join(args.checkpoint_file, 'ckpt-15'))  # sometimes the latest model is not the best,then use this

        #manager = tf.train.CheckpointManager(checkpoint, directory=args.checkpoint_file, max_to_keep=3)
        #checkpoint.restore(manager.latest_checkpoint)

        # if manager.latest_checkpoint:
        #     tf.print("Model restored successfully.")
        #
        #     # 打印模型参数
        #     for variable in model.variables:
        #         tf.print(variable.name, variable.shape)
        # else:
        #     tf.print("Error loading checkpoint.")

        if args.evaluation == 'evaluate_fl':
           #ds_test = ds_test.batch(64)
            evaluate_fl(model, ds_test)
        # elif args.evaluation == 'confusionmatrix':
        #     confusionmatrix(model, ds_test)
        # elif args.evaluation == 'Dimensionality_Reduction':
        #     Dimensionality_Reduction(model, ds_test)
        # elif args.evaluation == 'ROC':
        #     ROC(model, ds_test)


if __name__ == "__main__":
    app.run(main)
