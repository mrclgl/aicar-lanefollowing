
import os
import csv
import cv2
import shutil
import time

import numpy as np

from threading import Thread

date_filename = "data.csv"
steering_center_pulse = 1444.54

dataset_dir = "./dataset"
output_data_filename = "data.csv"
image_filename_base = "frame"

SAVE_FILE_EXT = ".png"

WIDTH = 256#292
HEIGHT = 128#164

THREADS = 8

MIN_SHADOWS = 2
MAX_SHADOWS = 5
MIN_SHADOW_POLY = 3
MAX_SHADOW_POLY = 8

MIN_SHADOW_STRENGTH = 0.4
MAX_SHADOW_STRENGTH = 0.6
MAX_BLUR_AMOUNT = 61

SHADOW_INVERT_CHANCE = 0.5

SHADOW_AUG_COUNT = 1

def generate_shadow_coordinates(imshape, no_of_shadows=1):
    vertices_list=[]
    for index in range(no_of_shadows):
        vertex=[]
        for dimensions in range(np.random.randint(MIN_SHADOW_POLY, MAX_SHADOW_POLY)): ## Dimensionality of the shadow polygon
            vertex.append((imshape[1]*np.random.uniform(), imshape[0]*np.random.uniform()))
            vertices = np.array([vertex], dtype=np.int32) ## single shadow vertices
            vertices_list.append(vertices)
    return vertices_list ## List of shadow vertices

def add_shadow(image,no_of_shadows=1):
    image_HLS = cv2.cvtColor(image,cv2.COLOR_BGR2HLS) ## Conversion to HLS
    mask = np.zeros_like(image, dtype=np.float32)
    imshape = image.shape
    vertices_list = generate_shadow_coordinates(imshape, no_of_shadows) #3 getting list of shadow vertices

    for vertices in vertices_list:
        cv2.fillPoly(mask, vertices, 1) ## adding all shadow polygons on empty mask, single 255 denotes only red channel

    blur_kernel_size = int(np.random.uniform()*MAX_BLUR_AMOUNT)
    if blur_kernel_size > 0 and blur_kernel_size % 2 == 0:
        blur_kernel_size += 1

    if blur_kernel_size > 0:
        mask = cv2.GaussianBlur(mask, (blur_kernel_size, blur_kernel_size), 0)

    if np.random.uniform() < SHADOW_INVERT_CHANCE:
        mask[:,:,0] = 1 - mask[:,:,0]

    shadow_strength = np.random.uniform(low=MIN_SHADOW_STRENGTH, high=MAX_SHADOW_STRENGTH)
    mask[:,:,0] = 1 - mask[:,:,0]*shadow_strength

    image_HLS[:,:,1] = image_HLS[:,:,1]*mask[:,:,0]   ## if red channel is hot, image's "Lightness" channel's brightness is lowered
    image_HLS[:,:,2] = image_HLS[:,:,2]*mask[:,:,0]   ## if red channel is hot, image's "Saturation" channel's brightness is lowered
    image_BGR = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2BGR) ## Conversion to RGB
    return image_BGR

def process_files(rows, start_index, stop_index, output_rows, progress, thread_index):

    for i in range(start_index, stop_index):

        progress[thread_index] = (i - start_index) / (stop_index - start_index)

        row = rows[i]
        
        path = row[0]
        filename, file_ext = os.path.splitext(path)
        throttle = int(row[1])
        steering = int(row[2])
        speed = float(row[3])

        img_orig = cv2.imread(path)
        img_orig = cv2.resize(img_orig, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        img_orig_flipped = cv2.flip(img_orig, 1)

        for aug_num in range(1 + SHADOW_AUG_COUNT):
            if aug_num > 0:
                img         = add_shadow(img_orig,         no_of_shadows=int(np.random.randint(low=MIN_SHADOWS, high=MAX_SHADOWS)))
                img_flipped = add_shadow(img_orig_flipped, no_of_shadows=int(np.random.randint(low=MIN_SHADOWS, high=MAX_SHADOWS)))
            else:
                img         = img_orig
                img_flipped = img_orig_flipped

            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #img_flipped = cv2.cvtColor(img_flipped, cv2.COLOR_BGR2GRAY)

            target_filename = os.path.join(dataset_dir, image_filename_base + str(i) + "-" + str(aug_num))

            target_filename_img = target_filename + SAVE_FILE_EXT
            target_filename_img_flipped = target_filename + "_flipped" + SAVE_FILE_EXT

            cv2.imwrite(target_filename_img, img)
            cv2.imwrite(target_filename_img_flipped, img_flipped)

            steering_flipped = int(steering_center_pulse + (steering_center_pulse - steering))

            output_rows[thread_index].append([target_filename_img, throttle, steering, speed])
            output_rows[thread_index].append([target_filename_img_flipped, throttle, steering_flipped, speed])

if __name__ == "__main__":

    if os.path.isdir(dataset_dir):
        shutil.rmtree(dataset_dir)

    os.makedirs(dataset_dir)

    csv_rows = []
    with open(date_filename, "r") as file:
        csv_file = csv.reader(file)
        csv_rows = list(csv_file)

    print("Found %d rows in data.csv...\n" % len(csv_rows))

    threading_count = len(csv_rows) // THREADS
    progress = [0 for i in range(THREADS)]
    output_rows = [[] for i in range(THREADS)]
    threads = []

    for i in range(THREADS):
        thread = Thread(target=process_files, args=(
            csv_rows,
            i * threading_count,
            (i + 1) * threading_count if (i + 1) < THREADS else len(csv_rows),
            output_rows,
            progress,
            i))

        thread.setDaemon(True)
        thread.start()
        threads.append(thread)

    while True:
        alive = False
        for thread in threads:
            if thread.is_alive():
                alive = True
                break

        print("Progress: %.2f%%" % (sum(progress) / len(progress) * 100))

        if alive == False:
            break

        time.sleep(1)

    print("\nWrote %d images" % (len(csv_rows) * (2 * (1 + SHADOW_AUG_COUNT))))

    with open(os.path.join(dataset_dir, output_data_filename), "w") as file:
        csv_writer = csv.writer(file)
        for row_list in output_rows:
            for row in row_list:
                csv_writer.writerow(row)

    print("Wrote %d rows to data.csv" % sum([len(row_list) for row_list in output_rows]))