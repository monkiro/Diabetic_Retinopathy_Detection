import gin
import tensorflow as tf
from keras import layers
# from architecture.layers import vgg_block
# from keras import regularizers
# from layers import res_stem, res_build_block, res_basic_block

@gin.configurable
def Basic_CNN(input_shape, base_filters, kernel_size, dense_units, dropout_rate, n_classes):
    """Defines a basic CNN Network as benchmark.
      in oder to learn the effects of different layers
        """
    inputs = tf.keras.Input(input_shape)
    out = tf.keras.layers.Conv2D(base_filters, kernel_size, padding='same', activation=tf.nn.relu)(inputs)
    out = tf.keras.layers.MaxPool2D((2, 2))(out)
    out = tf.keras.layers.Conv2D(base_filters * 2, kernel_size, padding='same', activation=tf.nn.relu)(out)
    out = layers.BatchNormalization()(out)
    out = tf.keras.layers.MaxPool2D((2, 2))(out)
    out = tf.keras.layers.Conv2D(base_filters * 4, kernel_size, padding='same', activation=tf.nn.relu)(out)
    out = tf.keras.layers.MaxPool2D((2, 2))(out)
    out = tf.keras.layers.Conv2D(base_filters * 8, kernel_size, padding='same', activation=tf.nn.relu)(out)
    out = layers.BatchNormalization()(out)
    out = tf.keras.layers.MaxPool2D((2, 2))(out)
    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dense(dense_units, activation=tf.nn.relu)(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    out = tf.keras.layers.Dense(dense_units/2, activation=tf.nn.relu)(out)
    out = tf.keras.layers.Dense(dense_units / 2, activation=tf.nn.relu)(out)
    outputs = tf.keras.layers.Dense(n_classes, activation=tf.nn.softmax)(out)
    # outputs = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(out)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='Basic_CNN')

