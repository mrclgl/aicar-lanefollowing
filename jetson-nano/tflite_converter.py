
import tensorflow as tf
import numpy as np
import argparse

from train import load_parsed_dataset
from train import process_parsed_dataset

from model import Model

WIDTH = 292
HEIGHT = 164
CHANNELS = 1

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

def representative_dataset_gen():
    dataset = load_parsed_dataset(args.records_path, args.records_ext, args.data_threads)
    dataset = dataset.shuffle(args.data_shuffle_size).batch(1).take(args.calibration_sample_size)
    index = 0
    for elem in dataset:
        if (index + 1) % (args.calibration_sample_size//10) == 0:
            print("Calibration %.2f%%" % ((index + 1) / args.calibration_sample_size * 100))
        # Get sample input data as a numpy array in a method of your choosing.
        yield [elem[0]["image"], elem[0]["speed"]]
        index += 1

def main(args):

    model = Model(1, WIDTH, HEIGHT, CHANNELS)

    build_model_dataset = load_parsed_dataset(args.records_path, args.records_ext, args.data_threads).batch(1)
    for data in build_model_dataset:
        model(data[0])
        break
    model.load_weights(args.model_weights)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
    converter.target_spec.supported_types = [tf.float16]
    converter.representative_dataset = representative_dataset_gen

    tflite_model = converter.convert()
    
    with open(args.tflite_model, "wb") as file:
        file.write(tflite_model)

if __name__ == "__main__":

    argparse = argparse.ArgumentParser()
    argparse.add_argument("-model_weights", default="./model_weights.h5", type=str)
    argparse.add_argument("-tflite_model", default="./model.tflite", type=str)
    argparse.add_argument("-records_path", default="./data/tfrecords", type=str)
    argparse.add_argument("-records_ext", default=".tfrecords", type=str)
    argparse.add_argument("-data_threads", default=16, type=int)
    argparse.add_argument("-data_shuffle_size", default=4096, type=int)
    argparse.add_argument("-calibration_sample_size", default=256, type=int)

    args = argparse.parse_args()

    main(args)
    