
import cv2
import numpy as np
import time

import tflite_runtime.interpreter as tflite

from threading import Thread

from aicar.rccontroller import *
from aicar.gps import GPS
from aicar.camera import Camera

WIDTH = 292
HEIGHT = 164

THROTTLE_MIN = 1300
THROTTLE_MAX = 1550

def per_image_standerdization(img):
    mean, std = cv2.meanStdDev(img)
    N = np.prod(img.shape)
    return (img - np.reshape(mean, (3))) / np.maximum(np.reshape(std, (3)), 1.0/np.sqrt(N))

if __name__ == "__main__":

    rc = RCController(0)

    if rc.change_mode_of_operation(ModeOfOperation.AI) == False:
        exit(1)

    print("- Loading TF-Lite interpreter")
    interpreter = tflite.Interpreter(model_path="./model.tflite")

    print(interpreter.get_input_details())
    print(interpreter.get_output_details())
    
    print("- Allocating interpreter tensors")
    interpreter.allocate_tensors()

    camera = Camera(
        capture_res=(1280, 720),
        capture_fps=60
        )

    gps = GPS()

    print("Waiting for data...")
    while camera.get_image() is None:
        time.sleep(0.1)

    input("\nAll systems ready! Press Enter to start...")

    try:
        while True:
            start_time = time.time()

            image = camera.get_image()
            
            gps_speed = gps.get_speed()
            gps_speed = np.reshape(gps_speed, (1, 1)).astype(np.float32)

            image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.expand_dims(image, axis=2)
            image = image.astype(np.float32)/255.0
            image = np.expand_dims(image, axis=0)

            interpreter.set_tensor(0, image)
            interpreter.set_tensor(1, gps_speed)
            interpreter.invoke()
            result = interpreter.get_tensor(68)[0]

            throttle = max(min(int(round(result[0])), THROTTLE_MAX), THROTTLE_MIN)
            steering = int(round(result[1]))
            fps = 1.0 / (time.time() - start_time)

            rc.set_rc_control_signals(throttle, steering)

            print("Throttle: %s - Steering: %s [FPS: %.2f]" % (str(throttle).ljust(4), str(steering).ljust(4), fps))

    except KeyboardInterrupt:
        rc.request_mode_of_operation(ModeOfOperation.Standby)

        print("\n\nDone!")

    camera.stop()
    gps.stop()


