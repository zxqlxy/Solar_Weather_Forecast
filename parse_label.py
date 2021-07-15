"""
Parse label file according to xrayfmt.txt
label is available at https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/solar-flares/x-rays/goes/xrs
for data untill 2017
"""

import csv
from datetime import datetime, timedelta
from collections import defaultdict

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

if __name__ == "__main__":
    name = "Hinode Flare Catalogue.csv"
    events = parse_label(name)
