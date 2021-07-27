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
import argparse


from parse_label import parse_label
from config import start, end, report_name, delta




base = "E:\\xl73\\data\\094\\"
start_time = datetime.strptime(start, '%Y-%m-%d %H:%M')
end_time = datetime.strptime(end, '%Y-%m-%d %H:%M')
dest = "E:\\xl73\\data\\average\\"
count = (end_time - start_time) // delta            # number of data points

targets = ['', '', '', 'C', 'M', "X"]

def process(events, args):
    """
    Pre-process the data
    """
    print(args)
    filename = 'maps_1024.npz'
    label_file = 'label.npz'
    if "256" in args:
        filename = 'maps_256.npz'

    src_list = np.zeros((3, 256, 256))
    tar_list = np.zeros((count,))
    temp = np.zeros((3, 1024, 1024))
    time = start_time
    data_index = 0
    label_index = 0

    while time < end_time:
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

        # Ignore non flare data
        if tar == 0:
            if args.ignore_non_flares:
                continue
            elif random.random() < args.ignore_percent:
                continue

        # if data_index < 0:
        #     data_index += 1
        #     continue

        # Read file
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
        temp[0] = ma.array(data1, mask=mask.mask, fill_value = -1).filled() # fillded with -1
        temp[1] = ma.array(data2, mask=mask.mask, fill_value = -1).filled() # fillded with -1
        temp[2] = ma.array(data3, mask=mask.mask, fill_value = -1).filled() # fillded with -1

        # put in the data
        if args.down == 'average':
            src_list= downsample_256(temp, 'average')
        else:
            src_list = downsample_256(temp)
        tar_list[data_index] = tar

        # ignore nan files
        if np.any(np.isnan(src_list)):
            print("WARNING: nan encountered at ", yr + mo + da + "_" + ho + mi, tar)
            continue

        # Save file one by one
        if tar == 1:
            # sunpy.io.write_file(dest + "flares\\" + "AIA"+ yr + mo + da + "_" + ho + mi + '.fits', src_list, smap094.meta)
            np.save(dest + "flares\\" + "AIA"+ yr + mo + da + "_" + ho + mi + '.fits', src_list)
        else:
            # sunpy.io.write_file(dest + "non-flares\\" + "AIA"+ yr + mo + da + "_" + ho + mi + '.fits', src_list, smap094.meta)
            np.save(dest + "non-flares\\" + "AIA"+ yr + mo + da + "_" + ho + mi + '.fits', src_list)

        # np.set_printoptions(threshold=np.inf)
        # print(src_list[data_index])
        # print(smap094.meta)
        # if "plot_original" in argv:
        #     scaled_map = sunpy.map.Map(src_list[data_index][:, 26:230, 26:230], smap094.meta)
        #     fig = plt.figure()
        #     plt.subplot(projection=scaled_map)
        #     scaled_map.plot(cmap=smap094.cmap)
        #     scaled_map.draw_limb()
        #     plt.colorbar()
        #     plt.show()
        # elif "plot_masked" in argv:
        #     scaled_map = sunpy.map.Map(smap094.data, smap094.meta, mask=mask.mask)
        #     fig = plt.figure()
        #     plt.subplot(projection=scaled_map)
        #     scaled_map.plot(cmap=smap094.cmap)
        #     scaled_map.draw_limb()
        #     plt.show()

        print(label[1], label[4], data_index)
        print(yr + mo + da + "_" + ho + mi, tar)

        data_index += 1

    # print(src_list.shape)
    # print(tar_list.shape)

    print(data_index, time)
    # Save file
    if args.save_file:
        print('Loaded: ', src_list.shape)
        savez_compressed(filename, src_list, tar_list)
        print('Saved dataset: ', filename)


# Downsample any 1024 x 1024 np.array to 256 x 256
def downsample_256(src, opt = 'max'):
    """Use max-pooling or average-pooling to down-sample 

    Args:
        src ([np.array]): array of size N, h, w

    Returns:
        [np.array]: data downsized to 256 * 256
    """

    N,h,w = src.shape
    if opt == 'max':
        res = np.amax([src[:, i//4::4, i%4::4] for i in range(16)], axis = 0)
    elif opt == 'average':
        res = np.mean([src[:, i//4::4, i%4::4] for i in range(16)], axis = 0)
    else:
        print("Option not implemented")
        return None
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
    parser = argparse.ArgumentParser(description='process')
    parser.add_argument('--down', type=str, default="average", help='One of two downsample algo: average, max')
    parser.add_argument('--ignore_non_flares', type=bool, default=False, help='Ignoring non flares')
    parser.add_argument('--ignore_percent', type=float, default=0.0, help='Ignoring some percent of non flares')
    parser.add_argument('--save_file', type=bool, default=False, help='Save as NPZ file')

    # parser.add_argument('--use_gpu', dest='use_gpu', action='store_true', default=True, help='use gpu')
    # parser.add_argument('--exp_name', type=str, default='cudnn_test', help='output file name')
    args = parser.parse_args()
    
    events = parse_label(report_name)
    process(events, args)
