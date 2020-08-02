
import os
import argparse
import csv
import shutil
import cv2
import time
import random

from threading import Thread

import tensorflow as tf
import numpy as np

def _bytes_feature(value):
	"""Returns a bytes_list from a string / byte."""
	if isinstance(value, type(tf.constant(0))):
		value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
	"""Returns a float_list from a float / double."""
	return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _float_list_feature(value):
	"""Returns a float_list from a float / double."""
	return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def create_example(data_csv_row):

	image_path = data_csv_row[0]
	throttle = int(data_csv_row[1])
	steering = int(data_csv_row[2])
	speed = float(data_csv_row[3])

	feature = {}

	img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
	if len(img.shape) > 2:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	else:
		img = np.expand_dims(img, axis=2)

	img = img.astype(np.float32)
	img /= 255.0

	feature["image"] = _float_list_feature(img.flatten())
	feature["throttle"] = _float_feature(float(throttle))
	feature["steering"] = _float_feature(float(steering))
	feature["speed"] = _float_feature(speed)

	return tf.train.Example(features=tf.train.Features(feature=feature))

def write_tfrecord_file(thread_index, start_index, stop_index, output_path, max_size, rows, progress):
	
	index = start_index
	max_size = max_size * 1000 * 1000#Megabytes to bytes

	while index < stop_index:

		filename = os.path.join(output_path, "data" + str(thread_index) + "-" + str(index) + ".tfrecords")
		with tf.io.TFRecordWriter(filename) as writer:

			while os.stat(filename).st_size < max_size:

				progress[thread_index] = (index - start_index) / (stop_index - start_index)

				example = create_example(rows[index])
				writer.write(example.SerializeToString())

				index += 1

				if index >= stop_index:
					break


def main(args):

	if os.path.isdir(args.output):
		shutil.rmtree(args.output)

	os.makedirs(args.output)

	print("Reading data file...")
	rows = []
	with open(os.path.join(args.input, args.input_data_filename), "r") as file:
		csv_file = csv.reader(file)
		rows = [row for row in csv_file]

	print("Found %d rows..." % len(rows))
	random.shuffle(rows)

	threading_count = len(rows)//args.threads
	threads = []
	progress = [0 for i in range(args.threads)]

	start_time = time.time()
	print("Starting %d worker threads..." % args.threads)

	for i in range(args.threads):
		thread = Thread(target=write_tfrecord_file, args=(
			i,
			threading_count * i,
			threading_count * (i + 1) if (i+1) < args.threads else len(rows),
			args.output,
			args.record_megabytes,
			rows,
			progress
		))
		thread.setDaemon(True)
		thread.start()
		threads.append(thread)

	print("Threads started!")

	while True:
		alive = False
		for t in threads:
			if t.isAlive():
				alive = True
				break
		
		if alive == False:
			break

		print("Progress %.2f%%" % (sum(progress)/len(progress)*100))

		time.sleep(1)

	stop_time = time.time()
	print("\n\nFinished writing %d examples in %.2f seconds!" % (len(rows), stop_time - start_time))

if __name__ == "__main__":	

	argparse = argparse.ArgumentParser()	

	argparse.add_argument("-input", default="./dataset", type=str)
	argparse.add_argument("-input_data_filename", default="data.csv", type=str)
	argparse.add_argument("-output", default="./tfrecords", type=str)
	argparse.add_argument("-threads", default=16, type=int)
	argparse.add_argument("-record_megabytes", default=128, type=int)	

	args = argparse.parse_args()
	main(args)