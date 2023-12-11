import gin

import tensorflow as tf
from tensorflow import keras





class ConvBNRelu(tf.keras.Model):
#this is a vgg block
    def __init__(self, filters, kernel_size=3, strides=1, padding='SAME', weight_decay=0.0005, droprate=0, drop=False):
        super(ConvBNRelu, self).__init__()
        self.drop = drop
        self.conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                        padding=padding, kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.dropout = keras.layers.Dropout(rate=droprate)

    def call(self, inputs, training=False):
        out = self.conv(inputs)
        out = self.batchnorm(out)
        out = tf.nn.relu(out)
        if self.drop:
            out = self.dropout(out)

        return out

@gin.configurable
class VGG16Model(tf.keras.Model):
    def __init__(self,input_shape):
        super(VGG16Model, self).__init__()
        self.inputs = tf.keras.Input(input_shape)
        self.conv1 = ConvBNRelu(filters=64, kernel_size=[3, 3])
        self.conv2 = ConvBNRelu(filters=64, kernel_size=[3, 3])
        self.maxPooling1 = keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv3 = ConvBNRelu(filters=128, kernel_size=[3, 3])
        self.conv4 = ConvBNRelu(filters=128, kernel_size=[3, 3])
        self.maxPooling2 = keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv5 = ConvBNRelu(filters=256, kernel_size=[3, 3])
        self.conv6 = ConvBNRelu(filters=256, kernel_size=[3, 3])
        self.conv7 = ConvBNRelu(filters=256, kernel_size=[3, 3])
        self.maxPooling3 = keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv11 = ConvBNRelu(filters=512, kernel_size=[3, 3])
        self.conv12 = ConvBNRelu(filters=512, kernel_size=[3, 3])
        self.conv13 = ConvBNRelu(filters=512, kernel_size=[3, 3])
        self.maxPooling5 = keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv14 = ConvBNRelu(filters=512, kernel_size=[3, 3])
        self.conv15 = ConvBNRelu(filters=512, kernel_size=[3, 3])
        self.conv16 = ConvBNRelu(filters=512, kernel_size=[3, 3])
        self.maxPooling6 = keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.flat = keras.layers.Flatten()
        self.dropout = keras.layers.Dropout(rate=0.5)
        self.dense1 = keras.layers.Dense(units=512,
                                         activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005))
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.dense2 = keras.layers.Dense(units=1, activation='sigmoid')

    def vgg(self, training=False):
        out = self.conv1(self.inputs)
        out = self.conv2(out)
        out = self.maxPooling1(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.maxPooling2(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.maxPooling3(out)
        out = self.conv11(out)
        out = self.conv12(out)
        out = self.conv13(out)
        out = self.maxPooling5(out)
        out = self.conv14(out)
        out = self.conv15(out)
        out = self.conv16(out)
        out = self.maxPooling6(out)
        out = self.dropout(out)
        out = self.flat(out)
        out = self.dense1(out)
        out = self.batchnorm(out)
        out = self.dropout(out)
        outputs = self.dense2(out)
        return tf.keras.Model(inputs=self.inputs, outputs=outputs, name='vgg')

