import gin
import tensorflow as tf
from keras import layers
from keras import regularizers
"""
    apply pretrained models, which are trained on big datasets as feature extractor and add FC layers as classifier
 """

@gin.configurable
def tl_ConvNeXtBase(input_shape, n_classes, dense_units, dropout_rate):
    backbone = tf.keras.applications.convnext.ConvNeXtBase(
        include_top=False,
        weights='imagenet',
        input_shape=(256, 256, 3)
    )
    backbone.trainable = False
    inputs = tf.keras.Input(shape=input_shape)
    out = backbone(inputs, training=False)
    out = layers.GlobalAvgPool2D()(out)
    out = layers.Dense(dense_units * 8, activation="relu", kernel_regularizer=regularizers.l1(0.01))(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    out = layers.Dense(dense_units, activation="relu", kernel_regularizer=regularizers.l1(0.01))(out)

    outputs = tf.keras.layers.Dense(n_classes, activation=tf.nn.softmax)(out)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name='tl_EfficientNetV2L')

@gin.configurable
def tl_EfficientNetV2L(input_shape, n_classes, dense_units, dropout_rate):
    backbone = tf.keras.applications.efficientnet_v2.EfficientNetV2L(
        include_top=False,
        weights='imagenet',
        input_shape=(256, 256, 3)
    )
    backbone.trainable = False
    inputs = tf.keras.Input(shape=input_shape)
    out = backbone(inputs, training=False)
    out = layers.GlobalAvgPool2D()(out)
    out = layers.Dense(dense_units * 8, activation="relu", kernel_regularizer=regularizers.l1(0.01))(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    out = layers.Dense(dense_units, activation="relu", kernel_regularizer=regularizers.l1(0.01))(out)

    outputs = tf.keras.layers.Dense(n_classes, activation=tf.nn.softmax)(out)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name='tl_EfficientNetV2L')



@gin.configurable
def tl_xception(input_shape, n_classes, dense_units, dropout_rate):
    backbone = tf.keras.applications.xception.Xception(
        include_top=False,
        weights='imagenet',
        input_shape=(256, 256, 3)
    )
    backbone.trainable = False
    inputs = tf.keras.Input(shape=input_shape)
    out = backbone(inputs, training=False)
    out = layers.GlobalAvgPool2D()(out)
    out = layers.Dense(dense_units * 8, activation="relu", kernel_regularizer=regularizers.l1(0.01))(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    out = layers.Dense(dense_units, activation="relu", kernel_regularizer=regularizers.l1(0.01))(out)

    outputs = tf.keras.layers.Dense(n_classes, activation=tf.nn.softmax)(out)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name='tl_xception')


@gin.configurable
def tl_inception_resnet(input_shape, n_classes, dense_units, dropout_rate):
    backbone = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(256, 256, 3)
    )
    backbone.trainable = False
    inputs = tf.keras.Input(shape=input_shape)
    out = backbone(inputs, training=False)
    out = layers.GlobalAvgPool2D()(out)
    out = layers.Dense(dense_units * 8, activation="relu", kernel_regularizer=regularizers.l1(0.01))(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    out = layers.Dense(dense_units, activation="relu", kernel_regularizer=regularizers.l1(0.01))(out)

    outputs = tf.keras.layers.Dense(n_classes, activation=tf.nn.softmax)(out)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name='tl_inception_resnet')

@gin.configurable
def tl_inception(input_shape, n_classes, dense_units, dropout_rate):
    backbone = tf.keras.applications.inception_v3.InceptionV3(
        include_top=False,
        weights='imagenet',
        input_shape=(256, 256, 3)
    )
    backbone.trainable = False
    inputs = tf.keras.Input(shape=input_shape)
    out = backbone(inputs, training=False)
    out = layers.GlobalAvgPool2D()(out)
    out = layers.Dense(dense_units * 8, activation="relu", kernel_regularizer=regularizers.l1(0.01))(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    out = layers.Dense(dense_units, activation="relu", kernel_regularizer=regularizers.l1(0.01))(out)

    outputs = tf.keras.layers.Dense(n_classes, activation=tf.nn.softmax)(out)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name='tl_inception')

@gin.configurable
def ResNet50(input_shape, n_classes, dense_units, dropout_rate):
    backbone = tf.keras.applications.resnet50.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(256, 256, 3)
    )
    backbone.trainable = False
    inputs = tf.keras.Input(shape=input_shape)
    out = backbone(inputs, training=False)
    out = layers.GlobalAvgPool2D()(out)
    out = layers.Dense(dense_units * 8, activation="relu", kernel_regularizer=regularizers.l1(0.01))(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    out = layers.Dense(dense_units, activation="relu", kernel_regularizer=regularizers.l1(0.01))(out)

    outputs = tf.keras.layers.Dense(n_classes, activation=tf.nn.softmax)(out)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name='ResNet50')