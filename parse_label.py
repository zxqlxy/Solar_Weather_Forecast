"""
Parse label file according to xrayfmt.txt
label is available at https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/solar-flares/x-rays/goes/xrs
for data untill 2017

Another Source:
Hinode: https://hinode.isee.nagoya-u.ac.jp/flare_catalogue/
"""

import csv
from datetime import datetime, timedelta
from collections import defaultdict
from os import path
import numpy as np


from config import start, end, report_name, delta
start_time = datetime.strptime(start, '%Y-%m-%d %H:%M')
end_time = datetime.strptime(end, '%Y-%m-%d %H:%M')

base = "E:\\xl73\\data\\average\\flares\\"


def parse_label(name:str) -> list[tuple]:
    f = open(name, 'r')
    csv_reader = csv.reader(f, delimiter=',')

    events = []
    header = next(csv_reader)
    print(header)

    dic = defaultdict(int)
    count = 0
    for line in csv_reader:
        start = datetime.strptime(line[1], '%Y/%m/%d %H:%M')
        peak = datetime.strptime(line[2], '%Y/%m/%d %H:%M')
        end = datetime.strptime(line[3], '%Y/%m/%d %H:%M')
        loc = line[4]
        clas = line[5][0]
        dic[clas] += 1
        intensity = float(line[5][1:])                  # range from 1.0-9.9
        tup = tuple([start, peak, end, loc, clas, intensity])
        events.append(tup)

    print(len(events), count, count/len(events))
    print(dic)
    return events[::-1]

def generate_yolo_label(events):
    dest = "E:\\xl73\\data\\average\\flares_label\\"

    R_SUN = 960.469 # Sun's radius
    # because we don't know how large it is, we can only assume a large box
    width, height = 0.25, 0.25

    time = start_time
    label_index = 0
    targets = ['C', 'M', 'X']
    while time < end_time:
        time += delta
        yr = str(time.year)
        mo = str(time.month) if time.month >= 10 else '0' + str(time.month)
        da = str(time.day) if time.day >= 10 else '0' + str(time.day)
        ho = str(time.hour) if time.hour >= 10 else '0' + str(time.hour)
        mi = str(time.minute) if time.minute >= 10 else '0' + str(time.minute)


        thisFile = "AIA"+ yr + mo + da + "_" + ho + mi + ".fits."

        # Skip non-existent or non-flare data
        if not path.isfile(base + thisFile + "npy"):
            # print("skip" + yr + mo + da + "_" + ho + mi)
            continue

        # Create label, N, C, M, X
        while True:
            # Reach the end
            if label_index == len(events):
                break

            label = events[label_index]
            dt = label[1]  # this is just the peak time
            
            # Need to move forward label data
            if time > dt:    
                label_index += 1
            # If find one break at once
            elif time < dt and time + delta > dt:
                # tar = targets.index(label[4]) if label[4] in targets else 0
                tar = 1 if label[4] in targets else 0
                label_index += 1
                break
            # Need to move forward the current time
            else:
                break
        
        if tar == 1:
            class_ = targets.index(label[4])
            x, y = 1, 1
            if label[3][0] == 'S':
                x = -1
            if label[3][3] == 'W':
                y = -1
            x = x * int(label[3][1:3])
            y = y * int(label[3][4:6])
            x_pix = (128 + R_SUN * np.sin(x/180*np.pi) / 9.6) / 256
            y_pix = (128 + R_SUN * np.sin(y/180*np.pi) / 9.6) / 256

            with open(dest + thisFile + "txt", "w") as f:
                f.write(str(class_) + " " + str(x_pix) + " " + str(y_pix) + " " + str(width) + " " + str(height))
                print(str(class_) + " " + str(x_pix) + " " + str(y_pix) + " " + str(width) + " " + str(height))


if __name__ == "__main__":
    name = "Hinode Flare Catalogue.csv"
    events = parse_label(name)
    generate_yolo_label(events)
