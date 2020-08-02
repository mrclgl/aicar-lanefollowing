
import os
import glob
import csv

output_filename = "data.csv"
image_file_name = "frame"
image_file_ext = ".png"

if __name__ == "__main__":

    data_csv_paths = glob.glob('./raw/**/data.csv', recursive=True)
    
    print("Found %d recording data files..." % len(data_csv_paths))

    with open(output_filename, "w") as file:
        file_csv = csv.writer(file)
        image_num = 0
        for path in data_csv_paths:
            dir_path = os.path.abspath(os.path.dirname(path))

            with open (path, "r") as in_data_file:
                in_data_csv = csv.reader(in_data_file)

                for row in in_data_csv:
                    image_num += 1
                    file_csv.writerow(
                        [os.path.join(dir_path, image_file_name + row[0] + image_file_ext),
                        row[1],
                        row[2],
                        row[3]])

    print("Wrote %d rows to %s " % (image_num, output_filename))

        