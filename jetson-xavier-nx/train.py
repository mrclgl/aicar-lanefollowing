
import argparse
import glob
import os
import cv2
import shutil

from tensorflow import keras
import tensorflow as tf
import tensorflow_addons as tfa

import numpy as np

import model

WIDTH = 256#292
HEIGHT = 128#164
CHANNELS = 3

BATCH_SIZE = 64
EPOCHS = 600

LEARNING_RATE = 0.0006
LEARNING_RATE_DECAY = 0.5
LEARNING_RATE_DECAY_EPOCHS = 100

SAMPLES = 124660
SHUFFLE_SIZE = 4096
CACHING = False

AUG_IMAGE_NOISE_AMOUNT = 0.1
AUG_IMAGE_ROTATION_ANGLE = 20

AUG_IMAGE_ZOOM_MAX = 0.1

AUG_IMAGE_CONTRAST_LOWER = 0.4
AUG_IMAGE_CONTRAST_UPPER = 1.6
AUG_IMAGE_BRIGHTNESS_MAX_DELTA = 0.2

AUG_IMAGE_SATURATION_LOWER = 0.4
AUG_IMAGE_SATURATION_UPPER = 1.3
AUG_IMAGE_HUE_MAX_DELTA = 0.1

TEST_SET_FRACTION = 0.1
VALIDATION_STEPS = 10

MODEL_FILE = "./model_weights.h5"

LOG_DIR = "./log"
LOG_UPDATE_FREQ = 1#Once per epoch

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

def create_empty_train_input():
    return { "image": np.zeros((BATCH_SIZE, HEIGHT, WIDTH, CHANNELS), dtype=np.float32), "speed": np.zeros((BATCH_SIZE, 1), dtype=np.float32) }

def parse_data(example_proto):

    data_feature_description = {
        "image": tf.io.FixedLenFeature([HEIGHT * WIDTH * CHANNELS], tf.float32),
        "throttle": tf.io.FixedLenFeature([1], tf.float32),
        "steering": tf.io.FixedLenFeature([1], tf.float32),
        "speed": tf.io.FixedLenFeature([1], tf.float32)
    }

    parsed = tf.io.parse_single_example(example_proto, data_feature_description)

    image = parsed["image"]
    image = tf.reshape(image, [HEIGHT, WIDTH, CHANNELS])

    throttle = parsed["throttle"]
    steering = parsed["steering"]
    speed = parsed["speed"]

    return { "image": image, "speed": speed }, [throttle, steering]

def augment_image(image):
    image = tf.image.random_brightness(image, AUG_IMAGE_BRIGHTNESS_MAX_DELTA)
    image = tf.image.random_contrast(image, lower=AUG_IMAGE_CONTRAST_LOWER, upper=AUG_IMAGE_CONTRAST_UPPER)
    if CHANNELS > 1:
        image = tf.image.random_saturation(image, lower=AUG_IMAGE_SATURATION_LOWER, upper=AUG_IMAGE_SATURATION_UPPER)
        image = tf.image.random_hue(image, AUG_IMAGE_HUE_MAX_DELTA)

    noise_amount = tf.random.uniform([1]) * AUG_IMAGE_NOISE_AMOUNT
    image += tf.random.uniform(image.shape, minval=-noise_amount, maxval=noise_amount)
    image = tf.clip_by_value(image, 0, 1)

    rot_angle = tf.random.uniform([1], minval=-1, maxval=1) * np.radians(AUG_IMAGE_ROTATION_ANGLE)
    image = tfa.image.rotate(image, rot_angle)

    zoom = tf.random.uniform([1])[0] * AUG_IMAGE_ZOOM_MAX
    image = tf.image.crop_and_resize(
        tf.expand_dims(image, axis=0),
        [[zoom, zoom, 1-zoom, 1-zoom]], 
        [0], 
        (HEIGHT, WIDTH))[0]

    return image

def process_parsed_dataset(parsed_dataset):

    dataset = parsed_dataset.repeat().shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE)
    dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

def load_parsed_dataset(records_path, records_ext, data_threads):

    tfrecord_files = []
    for file in glob.glob(os.path.join(records_path, "*" + records_ext)):
        tfrecord_files.append(file)

    raw_dataset = tf.data.TFRecordDataset(tfrecord_files)
    return raw_dataset.map(parse_data, num_parallel_calls=data_threads)

def take_at_indices(dataset, indices):

    result = None
    index = 0
    for i in sorted(indices):
        if i > 0:
            skip_num = i - index
            dataset = dataset.skip(skip_num)
            index += skip_num

        if result is None:
            result = dataset.take(1)
        else:
            result = result.concatenate(dataset.take(1))

    return result

def main(args):

    parsed_dataset = load_parsed_dataset(args.records_path, args.records_ext, args.data_threads)

    if CACHING:
        parsed_dataset = parsed_dataset.cache()

    if TEST_SET_FRACTION > 0.0:
        dataset_train = process_parsed_dataset(parsed_dataset.skip(TEST_SIZE))
        dataset_test = process_parsed_dataset(parsed_dataset.take(TEST_SIZE))
        validation_steps = VALIDATION_STEPS
    else:
        dataset_train = process_parsed_dataset(parsed_dataset)
        dataset_test = None
        validation_steps = None

    if args.visualize:
        vis_dataset = load_parsed_dataset(args.records_path, args.records_ext, args.data_threads)
        vis_dataset = take_at_indices(vis_dataset, VISUALIZE_EXAMPLE_INDICES)
        vis_dataset = vis_dataset.batch(1)#.repeat()
    else:
        vis_dataset = None

    show_dataset = None
    show_dataset_name = ""

    if args.show_train_data:
        show_dataset = dataset_train
        show_dataset_name = "train"
    elif args.show_test_data:
        show_dataset = dataset_test
        show_dataset_name = "test"
    elif args.show_vis_data:
        show_dataset = vis_dataset
        show_dataset_name = "visualize"

    if show_dataset is not None:
        for index, i in enumerate(show_dataset):
            print("\n[Example %d]" % index)
            print("Input: speed ", i[0]["speed"][0].numpy())
            print("Output: throttle %d - steering %d" % (i[1][0][0].numpy(), i[1][0][1].numpy()))
            img = i[0]["image"][0].numpy()
            if args.show_train_data:
                img = augment_image(img).numpy()
            cv2.imshow(show_dataset_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.waitKey()
        exit()

    if os.path.isdir(LOG_DIR):
        shutil.rmtree(LOG_DIR)

    model_obj = model.Model(BATCH_SIZE, WIDTH, HEIGHT, CHANNELS)

    if args.use_ckpt:
        model_obj(create_empty_train_input())
        model_obj.load_weights(MODEL_FILE)
        print("\nModel weights loaded!\n")

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        LEARNING_RATE,
        decay_steps=LEARNING_RATE_DECAY_STEPS,
        decay_rate=LEARNING_RATE_DECAY)

    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=LOG_DIR,
        update_freq=LOG_UPDATE_FREQ_STEPS,
        write_graph=True,
        write_images=True)
    tensorboard_model_logs_callback = model.TensorBoardModelCallback(
        model_obj, 
        lr_schedule, 
        STEPS_PER_EPOCH,
        MODEL_FILE,
        LOG_DIR)

    model_obj.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss=tf.losses.logcosh)

    history = model_obj.fit(
        x=dataset_train,
        validation_data=dataset_test,
        validation_steps=validation_steps,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        callbacks=[tensorboard_model_logs_callback, tensorboard_callback])

if __name__ == "__main__":

    argparse = argparse.ArgumentParser()

    argparse.add_argument("-records_path", default="./data/tfrecords", type=str)
    argparse.add_argument("-records_ext", default=".tfrecords", type=str)
    argparse.add_argument("-data_threads", default=8, type=int)
    argparse.add_argument("--show_train_data", action='store_true')
    argparse.add_argument("--show_test_data", action='store_true')
    argparse.add_argument("--show_vis_data", action='store_true')
    argparse.add_argument("--no_test", action='store_true')
    argparse.add_argument("--use_ckpt", action='store_true')
    argparse.add_argument("--visualize", action='store_true')

    args = argparse.parse_args()

    if args.no_test:
        TEST_SET_FRACTION = 0.0

    TEST_SIZE = int(SAMPLES * TEST_SET_FRACTION)
    STEPS_PER_EPOCH = (SAMPLES - TEST_SIZE) // BATCH_SIZE
    LEARNING_RATE_DECAY_STEPS = STEPS_PER_EPOCH * LEARNING_RATE_DECAY_EPOCHS
    LOG_UPDATE_FREQ_STEPS = STEPS_PER_EPOCH//LOG_UPDATE_FREQ

    main(args)