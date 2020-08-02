
import argparse
import time
import queue
import os
import shutil
import csv

import pynmea2
import serial
import io

from threading import Thread

import cv2

from aicar.rccontroller import ModeOfOperation
from aicar.rccontroller import RCController

from aicar.gps import GPS
from aicar.camera import Camera

FRAME_FILENAME = "frame"
FRAME_EXT = ".png"

RECORD_START_THROTTLE = 1550
RECORD_START_STEER_DELTA = 250

RECORD_MIN_DURATION = 5

RECORD_STOP_SPEED = 0.1
RECORD_STOP_SPEED_DURATION = 0.5

enc_running = False
enc_queue = queue.Queue()

def gstreamer_pipeline(
    recording_width=1280,
    recording_height=720,
    recording_framerate=60,
    flip_method=0,
    framerate=5
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! "
        "videorate ! video/x-raw,framerate=(fraction)%d/1 ! appsink"
        % (
            recording_width,
            recording_height,
            recording_framerate,
            flip_method,
            framerate
        )
    )

def encoder(path):
    global enc_running
    global enc_queue

    frame_index = 0
    while enc_running:
        while enc_queue.empty() == False:
            cv2.imwrite(os.path.join(path, FRAME_FILENAME + str(frame_index) + FRAME_EXT), enc_queue.get())
            frame_index += 1

        time.sleep(0.1)

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
    global enc_running
    global enc_queue

    rc = RCController(1)

    if rc.change_mode_of_operation(ModeOfOperation.RC) == False:
        exit(1)

    enc_thread = None
    gps = None
    camera = None

    try:
        session_path = os.path.join(args.output, args.sess_name)

        if os.path.isdir(session_path):
            print("\nSession with name \"%s\" already exists!" % args.sess_name)
            if ask_yes_or_no("Do you want to overwrite it?"):
                print("Deleting folder...")
                shutil.rmtree(session_path)
            else:
                exit(0)

        os.makedirs(session_path)

        gps = GPS()

        recording_index = 0

        while True:
            
            print("\n\nRecording: %d" % (recording_index + 1))
            print("Warming up...")

            recording_path = os.path.join(session_path, str(recording_index))
            os.makedirs(recording_path)

            enc_thread = Thread(target=encoder, args=(recording_path,))
            enc_running = True
            enc_thread.start()

            data_file = open(recording_path + "/data.csv", "w")
            data_writer = csv.writer(data_file)

            camera = Camera(
                capture_res=(1280, 720),
                capture_fps=60)
            
            warmup_start = time.time()
            countdown = int(args.warmup)
            while time.time() - warmup_start < args.warmup:
                time.sleep(0.1)
                if (args.warmup - (time.time() - warmup_start)) <= countdown:
                    print("%d..." % countdown)
                    countdown -= 1

            input("\nReady to record! Press Enter to confirm...")

            #steer = rc.get_rc_receiver_input()['steering']
            while rc.get_rc_receiver_input()['throttle'] < RECORD_START_THROTTLE:
            #while abs(steer - rc.get_rc_receiver_input()['steering']) < RECORD_START_STEER_DELTA:
                time.sleep(0.02)

            print("Recording started!")
            frame_index = 0
            frame_duration = 1.0 / float(args.fps)
            record_start_time = time.time()
            record_stop_time = -1
            while True:
                frame_start = time.time()

                image = camera.get_image()
                rc_input = rc.get_rc_receiver_input()
                gps_speed = gps.get_speed()

                data_writer.writerow([frame_index, rc_input['throttle'], rc_input['steering'], gps_speed])
                enc_queue.put(image)
                frame_index += 1

                if time.time() - record_start_time > RECORD_MIN_DURATION and gps_speed <= RECORD_STOP_SPEED:
                    if record_stop_time < 0:
                        record_stop_time = time.time()
                    elif time.time() - record_stop_time >= RECORD_STOP_SPEED_DURATION:
                        break
                else:
                    record_stop_time = -1

                time.sleep(max(frame_duration - (time.time() - frame_start), 0))

            print("Recording stopped! Saving files...")

            camera.stop()

            data_file.close()

            enc_running = False
            enc_thread.join()

            if ask_yes_or_no("\nDo you want to keep the recording?"):
                recording_index += 1
            else:
                print("Deleting...")
                shutil.rmtree(recording_path)
                
    except KeyboardInterrupt:
        rc.request_mode_of_operation(ModeOfOperation.Standby)

        if enc_thread is not None:
            enc_running = False
            enc_thread.join()

        if gps is not None:
            gps.stop()

        if camera is not None:
            camera.stop()
        
        print("\n\nDone")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Data Recorder for Lane-Following.")
    parser.add_argument("-output", default="./data_recorder", type=str)
    parser.add_argument("-sess_name", type=str, required=True)
    parser.add_argument("-warmup", type=float, default=5)
    parser.add_argument("-fps", type=int, default=15)

    args = parser.parse_args()
    main(args)
