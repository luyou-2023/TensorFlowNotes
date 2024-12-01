import os
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg16(tf.keras.Model):
    def __init__(self, vgg16_path=None):
        super(Vgg16, self).__init__()
        if vgg16_path is None:
            vgg16_path = os.path.join(os.getcwd(), "vgg16.npy")
            self.data_dict = np.load(vgg16_path, encoding='latin1', allow_pickle=True).item()

        # Initialize layers here
        self.conv1_1 = self.conv_layer("conv1_1")
        self.conv1_2 = self.conv_layer("conv1_2")
        self.pool1 = self.max_pool_2x2("pool1")

        self.conv2_1 = self.conv_layer("conv2_1")
        self.conv2_2 = self.conv_layer("conv2_2")
        self.pool2 = self.max_pool_2x2("pool2")

        self.conv3_1 = self.conv_layer("conv3_1")
        self.conv3_2 = self.conv_layer("conv3_2")
        self.conv3_3 = self.conv_layer("conv3_3")
        self.pool3 = self.max_pool_2x2("pool3")

        self.conv4_1 = self.conv_layer("conv4_1")
        self.conv4_2 = self.conv_layer("conv4_2")
        self.conv4_3 = self.conv_layer("conv4_3")
        self.pool4 = self.max_pool_2x2("pool4")

        self.conv5_1 = self.conv_layer("conv5_1")
        self.conv5_2 = self.conv_layer("conv5_2")
        self.conv5_3 = self.conv_layer("conv5_3")
        self.pool5 = self.max_pool_2x2("pool5")

        self.fc6 = self.fc_layer("fc6")
        self.relu6 = tf.keras.layers.ReLU()(self.fc6)

        self.fc7 = self.fc_layer("fc7")
        self.relu7 = tf.keras.layers.ReLU()(self.fc7)

        self.fc8 = self.fc_layer("fc8")
        self.prob = tf.nn.softmax(self.fc8, name="prob")

    def call(self, inputs, training=False):
        rgb_scaled = inputs * 255.0
        red, green, blue = tf.split(rgb_scaled, 3, 3)
        bgr = tf.concat([blue - VGG_MEAN[0], green - VGG_MEAN[1], red - VGG_MEAN[2]], 3)

        x = self.conv1_1(bgr)
        x = self.conv1_2(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.pool3(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.pool4(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.pool5(x)

        x = self.fc6(x)
        x = self.relu6(x)

        x = self.fc7(x)
        x = self.relu7(x)

        x = self.fc8(x)
        return self.prob

    def conv_layer(self, name):
        filters = self.get_conv_filter(name)
        conv = tf.keras.layers.Conv2D(filters=filters.shape[-1], kernel_size=(3, 3), strides=(1, 1), padding='same', name=name)(inputs)
        return conv

    def max_pool_2x2(self, name):
        return tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name=name)

    def fc_layer(self, name):
        return tf.keras.layers.Dense(self.data_dict[name][0].shape[-1], name=name)

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], dtype=tf.float32, name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], dtype=tf.float32, name="biases")
