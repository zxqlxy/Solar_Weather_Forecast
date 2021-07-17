# -*- coding: utf-8 -*-
"""
=================================
Process the EUV Data
=================================

The general workflow is generate ->
mask_out -> downsample -> add Data

(Optional show or save fig)

Total 6808
"""

import sunpy.io
import sunpy.map
import sunpy.data.sample
from sunpy.map.maputils import all_coordinates_from_map
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from numpy import savez_compressed
from datetime import datetime, timedelta
from os import path
import random

from parse_label import parse_label
from config import start, end, report_name, delta

base = "E:\\xl73\\data\\094\\"
start_time = datetime.strptime(start, '%Y-%m-%d %H:%M')
end_time = datetime.strptime(end, '%Y-%m-%d %H:%M')
locPath = "E:\\xl73\\data\\processed\\"
count = (end_time - start_time) // delta            # number of data points
count = 1808
targets = ['', '', '', 'C', 'M', "X"]

def process(events, *argv):
    """

    """
    print(argv)
    filename = 'maps_1024.npz'
    label_file = 'label.npz'
    if "256" in argv:
        filename = 'maps_256.npz'

    src_list = np.zeros((count, 3, 256, 256))
    tar_list = np.zeros((count,))
    temp = np.zeros((3, 1024, 1024))
    time = start_time
    data_index = -5000
    label_index = 0

    while time < end_time and data_index < count:
        time += delta
        yr = str(time.year)
        mo = str(time.month) if time.month >= 10 else '0' + str(time.month)
        da = str(time.day) if time.day >= 10 else '0' + str(time.day)
        ho = str(time.hour) if time.hour >= 10 else '0' + str(time.hour)
        mi = str(time.minute) if time.minute >= 10 else '0' + str(time.minute)

        tar = 0
        thisFile1 = "AIA"+ yr + mo + da + "_" + ho + mi + "_0094.fits"
        thisFile2 = "AIA"+ yr + mo + da + "_" + ho + mi + "_0171.fits"
        thisFile3 = "AIA"+ yr + mo + da + "_" + ho + mi + "_0304.fits"

        # Check if the file exists
        if not path.isfile(base + thisFile1) or not path.isfile(base + thisFile2) or not path.isfile(base + thisFile3):
            # impute using previous data 
            # src_list[data_index] = src_list[data_index - 1] # TODO
            # tar_list[data_index] = tar_list[data_index - 1]

            # impute using 0s
            # src_list[data_index] = np.zeros((3, 256, 256)) # TODO
            # tar_list[data_index] = 0
            # data_index += 1
            continue

        # Create label, N, C, M, X
        # future, it can include different classes and number of events and more
        
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
                tar = targets.index(label[4]) if label[4] in targets else 0
                label_index += 1
                break
            # Need to move forward the current time
            else:
                break

        # Ignore any non flare data
        if tar == 0:
            # data_index += 1
            continue
        
        if data_index < 0:
            data_index += 1
            continue
        # Ignore 75 percent
        # if random.random() > 0.25:
        #     continue


        # Once begine reading file, it's getting SLOWER
        # Make sure this reading is necessary
        # lis = sunpy.io.fits.read(base + thisFile)
        smap094 = sunpy.map.Map(base + thisFile1)
        smap171 = sunpy.map.Map(base + thisFile2)
        smap304 = sunpy.map.Map(base + thisFile3)

        # data shape is (1024, 1024)
        # data, header = lis[1]
        data1 = smap094.data
        data2 = smap171.data
        data3 = smap304.data

        # Start masking 
        hpc_coords = all_coordinates_from_map(smap094)
        r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / smap094.rsun_obs

        mask = ma.masked_greater_equal(r, 1)
        # In reality, everything smaller than 0 is wrong
        temp[0] = ma.array(data1, mask=mask.mask, fill_value = -1).filled() # fillded with -999
        temp[1] = ma.array(data2, mask=mask.mask, fill_value = -1).filled() # fillded with -999
        temp[2] = ma.array(data3, mask=mask.mask, fill_value = -1).filled() # fillded with -999

        # put in the data
        src_list[data_index] = downsample_256(temp)
        tar_list[data_index] = tar

        if "plot_original" in argv:
            scaled_map = sunpy.map.Map(smap094.data, smap094.meta)
            fig = plt.figure()
            plt.subplot(projection=scaled_map)
            scaled_map.plot(cmap=smap094.cmap)
            scaled_map.draw_limb()
            plt.show()
        elif "plot_masked" in argv:
            scaled_map = sunpy.map.Map(smap094.data, smap094.meta, mask=mask.mask)
            fig = plt.figure()
            plt.subplot(projection=scaled_map)
            scaled_map.plot(cmap=smap094.cmap)
            scaled_map.draw_limb()
            plt.show()

        print(label[1], label[4], data_index)
        print(yr + mo + da + "_" + ho + mi, tar)
        if np.any(np.isnan(src_list[data_index])):
            print("WARNING: nan encountered at ", data_index)
        data_index += 1

    print(src_list.shape)
    print(tar_list.shape)

    print(data_index, time)
    if "saveFile" in argv:
        print('Loaded: ', src_list.shape)
        savez_compressed(filename, src_list, tar_list)
        print('Saved dataset: ', filename)


# Downsample any 1024 x 1024 np.array to 256 x 256
def downsample_256(src):
    """Use max-pooling to down-sample

    Args:
        src ([np.array]): array of size N, h, w

    Returns:
        [np.array]: data downsized to 256 * 256
    """

    N,h,w = src.shape
    res = np.amax([src[:, i//4::4,i%4::4] for i in range(16)], axis = 0)
    return res


def plot_day(yr: str, mo: str, da: str):
    """

    :param yr:
    :param mo:
    :param da:
    """

    lis = sunpy.io.fits.read(
        base + "094/AIA" + yr + mo + da + "_0000_0094.fits")
    data, header = lis[1]
    print(header)
    # smap094 = sunpy.map.Map(
    #     base + "094/AIA" + yr + mo + da + "_0000_0094.fits")
    # im = smap094.plot()
    # plt.savefig(base + yr + mo + da + "_0000_0094.jpg")
    smap171 = sunpy.map.Map(
        base + "171/AIA" + yr + mo + da + "_0000_0171.fits")
    im = smap171.plot()
    # plt.savefig(base + yr + mo + da + "_0000_0171.jpg")
    smap193 = sunpy.map.Map(
        base + "193/AIA" + yr + mo + da + "_0000_0193.fits")
    im = smap193.plot()
    # plt.savefig(base + yr + mo + da + "_0000_0193.jpg")

    # Plot the high energy emission
    inside = (f * smap171.data) + ((1 - f) * smap193.data) / 116.54
    smap = 0.39 * (a1 * inside ** 1 + a2 * inside ** 2 + a3 * inside ** 3 + a4 * inside ** 4)
    tar = data - smap
    mymap = sunpy.map.Map(tar, header)
    fig, ax = plt.subplots()
    mymap.plot()
    ax.set_title('High Temperature Emission 2019-03-09')

    # plt.show()
    plt.savefig(base + yr + mo + da + "_Fe_XVIII.jpg")


if __name__ == "__main__":
    events = parse_label(report_name)
    # events = None
    process(events, "256", "saveFile")
    # plot_day("2019", "03", "09")
