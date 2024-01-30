import os
# import sys
# sys.path.append("./diabetic_retinopathy")
import argparse
import random
import gin
import logging
import numpy as np
from absl import app
import tensorflow as tf
from deep_visualization.Dimensionality_Reduction import dimensionality_reduction
from train import Trainer
from evaluation.evaluate_loss import evaluate0, evaluate_fl
from evaluation.metrics import confusionmatrix, ROC
from input_pipeline.dataset import load
from utils import save, logger
from model.basic_CNN import *
from model.vgg_like import *
from model.vgg import *
from model.resnet import *
from show_cam import deep_visualization
# from model.transferlearning import *



parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--model', choices=['Basic_CNN', 'vgg_like', 'vgg', 'resnet', 'tl_inception', 'tl_xception', 'tl_inception_resnet'
                                        'tl_ConvNeXtBase', 'tl_EfficientNetV2L', 'ResNet50' ],
                    default='Basic_CNN', help='choose model')
parser.add_argument('--mode', choices=['train', 'test'], default='train', help='train or test')
parser.add_argument('--evaluation', choices=['evaluate_fl', 'confusionmatrix', 'dimensionality_reduction', 'ROC',
                                             'deep_visualization', 'evaluate0'],
                        default='evaluate_fl', help='evaluation methods')
parser.add_argument('--checkpoint_file', type=str, default='D:\\DL_Lab_P1\\ckpts\\basic_cnn03\\',
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
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


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

    # input for deep visualization
    img_path = "D:\\DL_Lab_P1\\dataset_processed\\images\\showcam\\IDRiD_049.jpg"

    if args.model == 'Basic_CNN':
        model = Basic_CNN()
    elif args.model == 'vgg_like':
        model = vgg_like()
    elif args.model == 'vgg':
        model_instance = VGG16Model()
        model = model_instance.vgg()
    elif args.model == 'resnet':
        model = resnet(input_shape=ds_info.input_shape, n_classes=ds_info.n_classes)

    # elif args.model == 'tl_ConvNeXtBase':
    #     model = tl_ConvNeXtBase(input_shape=ds_info.input_shape, n_classes=ds_info.n_classes)
    # elif args.model == 'tl_EfficientNetV2L':
    #     model = tl_EfficientNetV2L(input_shape=ds_info.input_shape, n_classes=ds_info.n_classes)
    # elif args.model == 'tl_xception':
    #     model = tl_xception(input_shape=ds_info.input_shape, n_classes=ds_info.n_classes)
    # elif args.model == 'tl_inception_resnet':
    #     model = tl_inception_resnet(input_shape=ds_info.input_shape, n_classes=ds_info.n_classes)
    # elif args.model == 'tl_inception':
    #     model = tl_inception(input_shape=ds_info.input_shape, n_classes=ds_info.n_classes)
    # elif args.model == 'ResNet50':
    #     model = ResNet50(input_shape=ds_info.input_shape, n_classes=ds_info.n_classes)
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
        #     for variable in model.variables:
        #         tf.print(variable.name, variable.shape)
        # else:
        #     tf.print("Error loading checkpoint.")

        if args.evaluation == 'evaluate_fl':
            evaluate_fl(model, ds_test)
        elif args.evaluation == 'evaluate0':
            evaluate0(model, ds_test)
        elif args.evaluation == 'confusionmatrix':
            confusionmatrix(model, ds_test)
        elif args.evaluation == 'ROC':
            ROC(model, ds_test)
        elif args.evaluation == 'dimensionality_reduction':
            dimensionality_reduction(model, ds_test)
        elif args.evaluation == 'deep_visualization':
            deep_visualization(model, img_path)




if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # only need for deep_visualization
    # tf.config.set_visible_devices([], 'GPU')
    physical_devices = tf.config.list_physical_devices('GPU')
    print("Available GPUs:", physical_devices)     # Print the list of available GPUs
    app.run(main)
