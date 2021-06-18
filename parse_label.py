"""
Parse label file according to xrayfmt.txt
label is available at https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/solar-flares/x-rays/goes/xrs
for data untill 2017
"""

def parse_label(name:str) -> list[tuple]:
    f = open(name, 'r')
    lines = f.readlines()

    events = []

    for line in lines:
        year = line[5:7]
        month = line[7:9]
        day = line[9:11]
        start = line[13:17]
        end = line[18:22]
        peak = line[23:27]
        clas = line[59]
        intensity = line[60:63]                  # range from 1.0-9.9
        tup = tuple([year, month, day, start, end, peak, clas, intensity])
        events.append(tup)

    return events

if __name__ == "__main__":
    name = "goes-xrs-report_2000.txt"
    events = parse_label(name)
    print(events)