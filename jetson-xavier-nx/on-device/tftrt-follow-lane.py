
import cv2
import numpy as np
import time
import argparse

from aicar.rccontroller import *
from aicar.gps import GPS
from aicar.camera import Camera

import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

WIDTH = 256
HEIGHT = 128

THROTTLE_MIN = 1300
THROTTLE_MAX = 1600

SAVED_MODEL_DIR = "./build/saved_model_trt"

FPS_STEPS = 100

def ask_yes_or_no(question):
    answer = False
    while True:
        ans = input(question + " [y/n]:")
        if ans == 'y' or ans == 'Y':
            answer = True
            break
        elif ans == 'n' or ans == 'N':
            answer = False
            break
        else:
            print("Invalid answer! Please answer with \"y\" or \"n\"...")
    
    return answer

def main(args):

    rc = RCController(1)

    saved_model = tf.saved_model.load(SAVED_MODEL_DIR)
    graph_func = saved_model.signatures["serving_default"]

    camera = Camera(
        capture_res=(1280, 720),
        capture_fps=60
        )

    gps = GPS()

    print("\nWaiting for data...")
    while camera.get_image() is None:
        time.sleep(0.1)

    try:
        while True:
            if ask_yes_or_no("\nDo you want to continue?") == False:
                break

            while rc.get_mode_of_operation() != ModeOfOperation.AI:
                rc.change_mode_of_operation(ModeOfOperation.AI)

            try:
                timer = time.time()
                timer_step = 0
                while True:

                    image = camera.get_image()
                    
                    gps_speed = gps.get_speed()
                    gps_speed = np.reshape(gps_speed, (1, 1)).astype(np.float32)

                    cuImage = cv2.cuda_GpuMat()
                    cuImage.upload(image)

                    cuImage = cv2.cuda.resize(cuImage, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                    cuImage = cv2.cuda.cvtColor(cuImage, cv2.COLOR_BGR2RGB)

                    image = cuImage.download()
                    image = image.astype(np.float32)/255.0
                    image = np.expand_dims(image, axis=0)

                    image_tensor = tf.constant(image, dtype=tf.float32)
                    speed_tensor = tf.constant(gps_speed, dtype=tf.float32)

                    result = graph_func(image=image_tensor, speed=speed_tensor)
                    result = result["output_1"].numpy()[0]
                    
                    throttle = max(min(int(round(result[0])), THROTTLE_MAX), THROTTLE_MIN)
                    steering = int(round(result[1]))

                    rc.set_rc_control_signals(throttle, steering)

                    if timer_step >= FPS_STEPS:
                        fps = timer_step / (time.time() - timer)

                        print("\n[Average FPS (%d): %.2f]" % (FPS_STEPS, fps))
                        print("Throttle: %s - Steering: %s" % (str(throttle).ljust(4), str(steering).ljust(4)))

                        timer = time.time()
                        timer_step = 0

                    timer_step += 1

            except KeyboardInterrupt:
                rc.set_rc_control_signals(1500, 1500)
                print("\n\nDone!")

    except KeyboardInterrupt:
        print("\n\nDone!")
    
    rc.request_mode_of_operation(ModeOfOperation.Standby)
    camera.stop()
    gps.stop()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-saved_model_dir", default="./build/saved_model_trt", type=str)

    args = parser.parse_args()

    main(args)


