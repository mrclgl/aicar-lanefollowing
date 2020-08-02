
import os
import cv2

import tensorflow as tf
import numpy as np

import train

DROPOUT = 0.1

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

        #256x128
        self.conv1 = tf.keras.layers.Conv2D(filters=8, kernel_size=(5,5), strides=(1,1), padding="same", activation=tf.nn.leaky_relu)
        #256x128
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding=("same"), strides=(2,2))
        #128x64

        self.conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(5,5), strides=(1,1), padding="same", activation=tf.nn.leaky_relu)
        #128x64
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding=("same"), strides=(2,2))
        #64x32

        self.conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same", activation=tf.nn.leaky_relu)
        self.conv4 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same", activation=tf.nn.leaky_relu)
        #64x32
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding=("same"), strides=(2,2))
        #32x16

        self.conv5 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation=tf.nn.leaky_relu)
        self.conv6 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation=tf.nn.leaky_relu)
        #32x16
        self.pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding=("same"), strides=(2,2))
        #16x8

        self.conv7 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", activation=tf.nn.leaky_relu)
        self.conv8 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", activation=tf.nn.leaky_relu)
        #16x8
        self.pool5 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding=("same"), strides=(2,2))
        #8x4

        self.conv9  = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="valid", activation=tf.nn.leaky_relu)
        #6x2
        self.conv10 = tf.keras.layers.Conv2D(filters=256, kernel_size=(2,2), strides=(1,1), padding="valid", activation=tf.nn.leaky_relu)
        #5x1

        self.flat = tf.keras.layers.Flatten()

        self.dense1 = tf.keras.layers.Dense(512, activation=tf.nn.leaky_relu)
        self.dense2 = tf.keras.layers.Dense(512, activation=tf.nn.leaky_relu)
        self.dense3 = tf.keras.layers.Dense(512, activation=tf.nn.leaky_relu)
        self.dense4 = tf.keras.layers.Dense(512, activation=tf.nn.leaky_relu)
        self.dense5 = tf.keras.layers.Dense(512, activation=tf.nn.leaky_relu)
        self.dense6 = tf.keras.layers.Dense(512, activation=tf.nn.leaky_relu)
        self.dense9 = tf.keras.layers.Dense(2,   activation=None)

        self.dropout1 = tf.keras.layers.Dropout(DROPOUT)
        self.dropout2 = tf.keras.layers.Dropout(DROPOUT)
        self.dropout3 = tf.keras.layers.Dropout(DROPOUT)
        self.dropout4 = tf.keras.layers.Dropout(DROPOUT)
        self.dropout5 = tf.keras.layers.Dropout(DROPOUT)
        self.dropout6 = tf.keras.layers.Dropout(DROPOUT)

    def call(self, inputs, training=False):

        images = inputs["image"]
        speeds = inputs["speed"]

        images.set_shape((self.batch_size, images.shape[1], images.shape[2], self.channels))

        if training is True:
            images_unstacked = tf.unstack(images, axis=0)
            for i, image in enumerate(images_unstacked):
                images_unstacked[i] = train.augment_image(image)

            images = tf.stack(images_unstacked, axis=0)

        images = tf.image.per_image_standardization(images)

        speeds.set_shape((self.batch_size, 1))

        x = self.conv1(images)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool3(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool4(x)

        x = self.conv7(x)
        x = self.conv8(x)
        x = self.pool5(x)

        x = self.conv9(x)
        x = self.conv10(x)

        x = self.flat(x)
        x = tf.concat([speeds, x], axis=1)

        x = self.dense1(x)
        x = self.dropout1(x, training=training)

        x = self.dense2(x)
        x = self.dropout2(x, training=training)

        x = self.dense3(x)
        x = self.dropout3(x, training=training)

        x = self.dense4(x)
        x = self.dropout4(x, training=training)

        x = self.dense5(x)
        x = self.dropout5(x, training=training)

        x = self.dense6(x)
        x = self.dropout6(x, training=training)

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

        os.makedirs(self.learning_rate_log_dir)
        file = open(os.path.join(self.learning_rate_log_dir, "learning-rate.csv"), "w")
        file.close()

    def save_layer_image(self, img, rel_path):

        img = img.numpy()
        img = np.squeeze(img)
        img *= 255.0/img.max()

        cv2.imwrite(os.path.join(self.layers_log_dir, rel_path), img)

    def on_epoch_end(self, epoch, logs={}):
        steps = (epoch + 1) * self.steps_per_epoch

        with open(os.path.join(self.learning_rate_log_dir, "learning-rate.csv"), "a") as file:
            file.write(str(self.lr_schedule(steps).numpy()) + "\n")

        self.model.save_weights(self.model_file)