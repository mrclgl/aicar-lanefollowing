
import numpy as np
import argparse

from train import load_parsed_dataset
from model import Model

import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

WIDTH = 256
HEIGHT = 128
CHANNELS = 3

def representative_dataset_gen():
    dataset = load_parsed_dataset(args.records_path, args.records_ext, args.data_threads)
    dataset = dataset.batch(1).take(args.calibration_sample_size)
    index = 0
    for elem in dataset:
        if (index + 1) % (args.calibration_sample_size//10) == 0:
            print("Calibration %.2f%%" % ((index + 1) / args.calibration_sample_size * 100))
        # Get sample input data as a numpy array in a method of your choosing.
        yield [elem[0]["image"], elem[0]["speed"]]
        index += 1

def main(args):

    params = trt.TrtConversionParams(precision_mode="FP16", use_calibration=False)

    converter = trt.TrtGraphConverterV2(input_saved_model_dir=args.saved_model, conversion_params=params)
    converter.convert()
    converter.build(representative_dataset_gen)
    converter.save(args.saved_model_trt)

if __name__ == "__main__":

    argparse = argparse.ArgumentParser()
    argparse.add_argument("-saved_model", default="./saved_model", type=str)
    argparse.add_argument("-saved_model_trt", default="./saved_model_trt", type=str)
    argparse.add_argument("-records_path", default="./data/tfrecords", type=str)
    argparse.add_argument("-records_ext", default=".tfrecords", type=str)
    argparse.add_argument("-data_threads", default=4, type=int)
    argparse.add_argument("-calibration_sample_size", default=2608, type=int)

    args = argparse.parse_args()

    main(args)
    
