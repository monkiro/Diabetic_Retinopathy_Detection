import gin
import logging
import tensorflow as tf
import tensorflow_datasets as tfds

from input_pipeline.image_pre import preprocess, augment

class DatasetInfo:
    # DatasetInfo class is used to store information about the dataset. it holds information about the dataset and
    # model architecture parameters
    input_shape = (32, 256, 256, 3)   # 32 images in a batch, each 256x256 pixels with 3 color channels (RGB).
    n_classes = 2
    fc_units = 32   # fully connected layer with 32 units
    filters_num = 32   # 32 filters in the first convolutional layer
    dropout_rate = 0.3   # prevent overfitting
    layer_dim = (1, 1, 1, 1)   # 1x1 convolutional layers

    def __init__(self, input_shape, n_classes, fc_units, filters_num, dropout_rate, layer_dim):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.fc_units = fc_units
        self.filters_num = filters_num
        self.dropout_rate = dropout_rate
        self.layer_dim = layer_dim


DatasetInfo = DatasetInfo((256, 256, 3), 2, 32, 32, 0.3, (1, 1, 1, 1))

def read_labeled_tfrecord(example):
    # read data from a TFRecord file in TensorFlow

    # It defines a dictionary with the feature name as key and a FixedLenFeature as value.
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),  # tf.string means bytestring
        "label": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element
    }

    # parse a single example into image and label tensors
    example = tf.io.parse_single_example(example, features)
    image = tf.io.decode_jpeg(example['image'], channels=3) / 255 # decode JPEG-encoded image to uint8 tensor
    label = example['label']

    return image, label  # returns a dataset of (image, label) pairs


def get_dataset(filenames):
    #  read data from TFRecord files and create a tf.data.Dataset object that is
    #  suitable for feeding into a TensorFlow model during training or inference.

    dataset = tf.data.TFRecordDataset(filenames)  # automatically interleaves reads from multiple files
    AUTOTUNE = tf.data.experimental.AUTOTUNE  # automatically tunes the value dynamically at runtime
    dataset = dataset.map(read_labeled_tfrecord, num_parallel_calls=AUTOTUNE)
    # parallelize the map transformation across multiple CPU cores

    return dataset
    # At this point, the dataset is a collection of (image, label) pairs with images that have been decoded and labels
    # that have been extracted. This dataset is now ready to be further batched, shuffled, and used for training or
    # evaluation in a machine learning model.






@gin.configurable
def load(name, data_dir):
    if name == "idrid":
        logging.info(f"Preparing dataset {name}...")
        ds_test = get_dataset(data_dir + 'test.tfrecords')
        ds_train = get_dataset(data_dir + 'train.tfrecords')
        ds_val = get_dataset(data_dir + 'validation.tfrecords')
        ds_info = DatasetInfo

        return prepare(ds_train, ds_val, ds_test, ds_info)

    elif name == "eyepacs":
        logging.info(f"Preparing dataset {name}...")
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            'diabetic_retinopathy_detection/btgraham-300',
            split=['train', 'validation', 'test'],
            shuffle_files=True,
            with_info=True,
            data_dir=data_dir
        )

        def _preprocess(img_label_dict):
            return img_label_dict['image'], img_label_dict['label']

        ds_train = ds_train.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_val = ds_val.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return prepare(ds_train, ds_val, ds_test, ds_info)

    elif name == "mnist":
        logging.info(f"Preparing dataset {name}...")
        (ds_train, ds_val, ds_test), ds_info = tf.keras.datasets.mnist.load_data(
            'mnist',
            split=['train[:90%]', 'train[90%:]', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
            data_dir=data_dir
        )

        return prepare(ds_train, ds_val, ds_test, ds_info)

    else:
        raise ValueError

@gin.configurable
def prepare(ds_train, ds_val, ds_test, ds_info, batch_size, caching):

    # It is designed to further process the datasets for training, validation, and testing phases of a machine
    # learning model.

    # Prepare training dataset
    ds_train = ds_train.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if caching:
        ds_train = ds_train.cache()
    ds_train = ds_train.map(
        augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples // 10)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.repeat(-1)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare validation dataset
    ds_val = ds_val.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.batch(batch_size)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare test dataset
    ds_test = ds_test.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_val, ds_test, ds_info