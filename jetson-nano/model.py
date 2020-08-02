
import os
import csv
import cv2

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

import train

class Model(tf.keras.Model):

    def image_grid(self, x, size_x=6, size_y=6):
        w = x.shape[1]
        h = x.shape[0]
        t = tf.unstack(x, axis=2)
        rows = [tf.concat(t[i*size_y:(i+1)*size_y], axis=0) 
                for i in range(size_x)]
        image = tf.concat(rows, axis=1)
        return tf.reshape(image, (1, h * size_y, w * size_x, 1))

    def __init__(self, batch_size, width, height, channels):
        super(Model, self).__init__()

        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.channels = channels

        self.train_augmentation = True

        #292x164
        self.conv1 = tf.keras.layers.Conv2D(filters=6, kernel_size=(5,5), strides=(1,1), padding="valid", activation=tf.nn.leaky_relu)
        #288x160
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding=("same"), strides=(2,2))
        #144x80

        self.conv2 = tf.keras.layers.Conv2D(filters=12, kernel_size=(5,5), strides=(1,1), padding="valid", activation=tf.nn.leaky_relu)
        #140x76
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding=("same"), strides=(2,2))
        #70x38

        self.conv3 = tf.keras.layers.Conv2D(filters=24, kernel_size=(3,3), strides=(1,1), padding="valid", activation=tf.nn.leaky_relu)
        #68x36
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding=("same"), strides=(2,2))
        #34x18

        self.conv4 = tf.keras.layers.Conv2D(filters=48, kernel_size=(3,3), strides=(1,1), padding="valid", activation=tf.nn.leaky_relu)
        #32x16
        self.pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding=("same"), strides=(2,2))
        #16x8

        self.conv5 = tf.keras.layers.Conv2D(filters=96, kernel_size=(3,3), strides=(1,1), padding="valid", activation=tf.nn.leaky_relu)
        #14x6
        self.pool5 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding=("same"), strides=(2,2))
        #7x3

        self.conv6 = tf.keras.layers.Conv2D(filters=192, kernel_size=(3,3), strides=(1,1), padding="valid", activation=tf.nn.leaky_relu)
        #5x1

        self.flat = tf.keras.layers.Flatten()

        self.dense1 = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)
        self.dense2 = tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu)
        self.dense3 = tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu)
        self.dense4 = tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu)
        self.dense5 = tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu)
        self.dense6 = tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu)
        self.dense7 = tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu)
        self.dense8 = tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu)
        self.dense9 = tf.keras.layers.Dense(2,   activation=None)

        self.layer1    = tf.Variable(tf.zeros([80, 144, 6]),   dtype=tf.float32, name='layer1', trainable=False)
        self.layer2    = tf.Variable(tf.zeros([38, 70, 12]),   dtype=tf.float32, name='layer2', trainable=False)
        self.layer3    = tf.Variable(tf.zeros([18, 34, 24]),   dtype=tf.float32, name='layer3', trainable=False)
        self.layer4    = tf.Variable(tf.zeros([8, 16, 48]),    dtype=tf.float32, name='layer4', trainable=False)
        self.layer5    = tf.Variable(tf.zeros([3, 7, 96]),     dtype=tf.float32, name='layer5', trainable=False)
        self.layer6    = tf.Variable(tf.zeros([1, 5, 192]),    dtype=tf.float32, name='layer6', trainable=False)

        self.layer_dense1 = tf.Variable(tf.zeros([1, 1, 128]), dtype=tf.float32, name='layer_dense1', trainable=False)
        self.layer_dense2 = tf.Variable(tf.zeros([1, 1, 64]),  dtype=tf.float32, name='layer_dense2', trainable=False)
        self.layer_dense3 = tf.Variable(tf.zeros([1, 1, 64]),  dtype=tf.float32, name='layer_dense3', trainable=False)
        self.layer_dense4 = tf.Variable(tf.zeros([1, 1, 64]),  dtype=tf.float32, name='layer_dense4', trainable=False)
        self.layer_dense5 = tf.Variable(tf.zeros([1, 1, 64]),  dtype=tf.float32, name='layer_dense5', trainable=False)
        self.layer_dense6 = tf.Variable(tf.zeros([1, 1, 64]),  dtype=tf.float32, name='layer_dense6', trainable=False)
        self.layer_dense7 = tf.Variable(tf.zeros([1, 1, 64]),  dtype=tf.float32, name='layer_dense7', trainable=False)
        self.layer_dense8 = tf.Variable(tf.zeros([1, 1, 64]),  dtype=tf.float32, name='layer_dense8', trainable=False)

    def call(self, inputs, training=False):

        images = inputs["image"]
        speeds = inputs["speed"]

        images.set_shape((self.batch_size, images.shape[1], images.shape[2], self.channels))

        if training is True and self.train_augmentation is True:
            images_unstacked = tf.unstack(images, axis=0)
            for i, image in enumerate(images_unstacked):
                images_unstacked[i] = train.augment_image(image)

            images = tf.stack(images_unstacked, axis=0)

        speeds.set_shape((self.batch_size, 1))

        x = self.conv1(images)
        x = self.pool1(x)
        if training is True:
            self.layer1.assign(x[0])

        x = self.conv2(x)
        x = self.pool2(x)
        if training is True:
            self.layer2.assign(x[0])

        x = self.conv3(x)
        x = self.pool3(x)
        if training is True:
            self.layer3.assign(x[0])

        x = self.conv4(x)
        x = self.pool4(x)
        if training is True:
            self.layer4.assign(x[0])

        x = self.conv5(x)
        x = self.pool5(x)
        if training is True:
            self.layer5.assign(x[0])

        x = self.conv6(x)
        if training is True:
            self.layer6.assign(x[0])

        x = self.flat(x)
        x = tf.concat([speeds, x], axis=1)

        x = self.dense1(x)
        if training is True:
            self.layer_dense1.assign(tf.reshape(x[0], self.layer_dense1.shape))

        x = self.dense2(x)
        if training is True:
            self.layer_dense2.assign(tf.reshape(x[0], self.layer_dense2.shape))

        x = self.dense3(x)
        if training is True:
            self.layer_dense3.assign(tf.reshape(x[0], self.layer_dense3.shape))

        x = self.dense4(x)
        if training is True:
            self.layer_dense4.assign(tf.reshape(x[0], self.layer_dense4.shape))

        x = self.dense5(x)
        if training is True:
            self.layer_dense5.assign(tf.reshape(x[0], self.layer_dense5.shape))

        x = self.dense6(x)
        if training is True:
            self.layer_dense6.assign(tf.reshape(x[0], self.layer_dense6.shape))

        x = self.dense7(x)
        if training is True:
            self.layer_dense7.assign(tf.reshape(x[0], self.layer_dense7.shape))

        x = self.dense8(x)
        if training is True:
            self.layer_dense8.assign(tf.reshape(x[0], self.layer_dense8.shape))

        x = self.dense9(x)

        return x

class TensorBoardModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, model:Model, lr_schedule, steps_per_epoch, model_file, log_dir):
        super().__init__()
        self.model = model
        self.lr_schedule = lr_schedule
        self.steps_per_epoch = steps_per_epoch
        self.model_file = model_file

        self.layers_log_dir = os.path.join(log_dir, "layers")
        self.learning_rate_log_dir = os.path.join(log_dir, "learning_rate")

        os.makedirs(self.layers_log_dir)

        os.makedirs(os.path.join(self.layers_log_dir, "conv-layer-1"))
        os.makedirs(os.path.join(self.layers_log_dir, "conv-layer-2"))
        os.makedirs(os.path.join(self.layers_log_dir, "conv-layer-3"))
        os.makedirs(os.path.join(self.layers_log_dir, "conv-layer-4"))
        os.makedirs(os.path.join(self.layers_log_dir, "conv-layer-5"))
        os.makedirs(os.path.join(self.layers_log_dir, "conv-layer-6"))

        os.makedirs(os.path.join(self.layers_log_dir, "conv-layer-dense-1"))
        os.makedirs(os.path.join(self.layers_log_dir, "conv-layer-dense-2"))
        os.makedirs(os.path.join(self.layers_log_dir, "conv-layer-dense-3"))
        os.makedirs(os.path.join(self.layers_log_dir, "conv-layer-dense-4"))
        os.makedirs(os.path.join(self.layers_log_dir, "conv-layer-dense-5"))
        os.makedirs(os.path.join(self.layers_log_dir, "conv-layer-dense-6"))
        os.makedirs(os.path.join(self.layers_log_dir, "conv-layer-dense-7"))
        os.makedirs(os.path.join(self.layers_log_dir, "conv-layer-dense-8"))

        os.makedirs(self.learning_rate_log_dir)
        file = open(os.path.join(self.learning_rate_log_dir, "learning-rate.csv"), "w")
        file.close()

    def save_layer_image(self, img, path):

        img = img.numpy()
        img = np.squeeze(img)
        img *= 255.0/img.max()

        cv2.imwrite(path, img)

    def on_epoch_end(self, epoch, logs={}):
        steps = (epoch + 1) * self.steps_per_epoch

        with open(os.path.join(self.learning_rate_log_dir, "learning-rate.csv"), "a") as file:
            file.write(str(self.lr_schedule(steps).numpy()) + "\n")

        self.save_layer_image(self.model.image_grid(self.model.layer1, size_x=3, size_y=2),   "conv-layer-1/epoch" + str(epoch) + ".png")
        self.save_layer_image(self.model.image_grid(self.model.layer2, size_x=4, size_y=3),   "conv-layer-2/epoch" + str(epoch) + ".png")
        self.save_layer_image(self.model.image_grid(self.model.layer3, size_x=8, size_y=3),   "conv-layer-3/epoch" + str(epoch) + ".png")
        self.save_layer_image(self.model.image_grid(self.model.layer4, size_x=8, size_y=6),   "conv-layer-4/epoch" + str(epoch) + ".png")
        self.save_layer_image(self.model.image_grid(self.model.layer5, size_x=16, size_y=6),  "conv-layer-5/epoch" + str(epoch) + ".png")
        self.save_layer_image(self.model.image_grid(self.model.layer6, size_x=16, size_y=12), "conv-layer-6/epoch" + str(epoch) + ".png")

        self.save_layer_image(self.model.image_grid(self.model.layer_dense1, size_x=16, size_y=8), "conv-layer-dense-1/epoch" + str(epoch) + ".png")
        self.save_layer_image(self.model.image_grid(self.model.layer_dense2, size_x=8, size_y=8),  "conv-layer-dense-2/epoch" + str(epoch) + ".png")
        self.save_layer_image(self.model.image_grid(self.model.layer_dense3, size_x=8, size_y=8),  "conv-layer-dense-3/epoch" + str(epoch) + ".png")
        self.save_layer_image(self.model.image_grid(self.model.layer_dense4, size_x=8, size_y=8),  "conv-layer-dense-4/epoch" + str(epoch) + ".png")
        self.save_layer_image(self.model.image_grid(self.model.layer_dense5, size_x=8, size_y=8),  "conv-layer-dense-5/epoch" + str(epoch) + ".png")
        self.save_layer_image(self.model.image_grid(self.model.layer_dense6, size_x=8, size_y=8),  "conv-layer-dense-6/epoch" + str(epoch) + ".png")
        self.save_layer_image(self.model.image_grid(self.model.layer_dense7, size_x=8, size_y=8),  "conv-layer-dense-7/epoch" + str(epoch) + ".png")
        self.save_layer_image(self.model.image_grid(self.model.layer_dense8, size_x=8, size_y=8),  "conv-layer-dense-8/epoch" + str(epoch) + ".png")

        self.model.save_weights(self.model_file)