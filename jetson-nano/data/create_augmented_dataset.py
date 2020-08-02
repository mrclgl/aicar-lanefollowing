
import os
import csv
import cv2
import shutil
import time

from threading import Thread

date_filename = "data.csv"
steering_center_pulse = 1444.54

dataset_dir = "./dataset"
output_data_filename = "data.csv"
image_filename_base = "frame"

WIDTH = 292
HEIGHT = 164

THREADS = 8

def process_files(rows, start_index, stop_index, output_rows, progress, thread_index):

    for i in range(start_index, stop_index):

        progress[thread_index] = (i - start_index) / (stop_index - start_index)

        row = rows[i]
        
        path = row[0]
        filename, file_ext = os.path.splitext(path)
        throttle = int(row[1])
        steering = int(row[2])
        speed = float(row[3])

        img = cv2.imread(path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)

        img_flipped = cv2.flip(img, 1)

        target_filename = os.path.join(dataset_dir, image_filename_base + str(i))

        target_filename_img = target_filename + file_ext
        target_filename_img_flipped = target_filename + "_flipped" + file_ext

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

    print("\nWrote %d images" % (len(csv_rows) * 2))

    with open(os.path.join(dataset_dir, output_data_filename), "w") as file:
        csv_writer = csv.writer(file)
        for row_list in output_rows:
            for row in row_list:
                csv_writer.writerow(row)

    print("Wrote %d rows to data.csv" % sum([len(row_list) for row_list in output_rows]))