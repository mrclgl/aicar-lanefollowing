
import argparse
import os
import csv

import cv2
import numpy as np

DATA_FILE = "data.csv"
DATA_FILE_BACKUP = "data_orig.csv"

IMAGE_BASE_NAME = "frame"
IMAGE_EXT = ".png"

def main(args):

    dirs = sorted(os.listdir(args.path))
    for dir_index, directory in enumerate(dirs):
        path = os.path.join(args.path, directory)

        data_rows = []
        with open(os.path.join(path, DATA_FILE)) as data_file:
            data_csv = csv.reader(data_file)
            data_rows = list(data_csv)

        if len(data_rows) == 0:
            continue

        print("\n\nRecording %d/%d:" % (dir_index + 1, len(dirs)))

        index = len(data_rows) - 1
        last_index = -1
        while True:

            img_path = os.path.join(path, IMAGE_BASE_NAME + data_rows[index][0] + IMAGE_EXT)
            img = cv2.imread(img_path)
            cv2.imshow("Display", img)

            if last_index != index:
                last_index = index
                print("\nFrame: %d - Throttle: %d - Steering: %d - Speed: %.2f" % (
                    int(data_rows[index][0]),
                    int(data_rows[index][1]),
                    int(data_rows[index][2]),
                    float(data_rows[index][3]),
                ))

            key = cv2.waitKeyEx(100)
            if key == 65361:
                index = max(0, index - 1)
            elif key == 65363:
                index = min(len(data_rows) - 1, index + 1)
            elif key == 13:
                if os.path.isfile(os.path.join(path, DATA_FILE_BACKUP)):
                    print("\nData backup file already exists!")
                    break

                os.rename(os.path.join(path, DATA_FILE), os.path.join(path, DATA_FILE_BACKUP))
                with open(os.path.join(path, DATA_FILE), "w") as file:
                    csv_writer = csv.writer(file)
                    for row in data_rows[:(index+1)]:
                        csv_writer.writerow(row)
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("path", type=str)

    args = parser.parse_args()
    main(args)