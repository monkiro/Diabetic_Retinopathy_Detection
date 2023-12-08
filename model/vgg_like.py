import gin
import tensorflow as tf
from keras import layers

# from keras import regularizers
# from layers import res_stem, res_build_block, res_basic_block


@gin.configurable
def vgg_like(input_shape, n_classes, base_filters, n_blocks, dense_units, dropout_rate):
    """Defines a VGG-like architecture.
    Parameters:
        input_shape (tuple: 3): input shape of the neural network
        n_classes (int): number of classes, corresponding to the number of output neurons
        base_filters (int): number of base filters, which are doubled for every VGG block
        n_blocks (int): number of VGG blocks
        dense_units (int): number of dense units
        dropout_rate (float): dropout rate
    Returns:
        (keras.Model): keras model object
    """
    assert n_blocks > 0, 'Number of blocks has to be at least 1.'

    inputs = tf.keras.Input(input_shape)
    out = vgg_block(inputs, base_filters)
    for i in range(2, n_blocks):
        out = vgg_block(out, base_filters * 2 ** (i))
    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dense(dense_units, activation=tf.nn.relu)(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(out)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='vgg_like')


@gin.configurable
def vgg_block(inputs, filters, kernel_size):
    """A single VGG block consisting of two convolutional layers, followed by a max-pooling layer.
    Parameters:
        inputs (Tensor): input of the VGG block
        filters (int): number of filters used for the convolutional layers
        kernel_size (tuple: 2): kernel size used for the convolutional layers, e.g. (3, 3)
    Returns:
        (Tensor): output of the VGG block
    """
    out = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation=tf.nn.relu)(inputs)
    out = layers.BatchNormalization()(out)
    out = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation=tf.nn.relu)(out)
    out = layers.BatchNormalization()(out)
    out = tf.keras.layers.MaxPool2D((2, 2))(out)

    return out