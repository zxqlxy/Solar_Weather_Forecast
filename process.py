# -*- coding: utf-8 -*-
"""
=================================
Process the EUV Data
=================================

The general workflow is generate ->
mask_out -> downsample -> add Data

(Optional show or save fig)
"""

import sunpy.io
import sunpy.map
import sunpy.data.sample
from sunpy.map.maputils import all_coordinates_from_map
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from numpy import savez_compressed, asarray
from datetime import datetime, timedelta
from os import path


from parse_label import parse_label
from config import start, end, report_name, delta

base = "/Users/lxy/Desktop/Rice/Su 21/Solar_Weather_Forecast/data/SDOAIA/"
start_time = datetime.strptime(start, '%Y-%m-%d %H:%M')
end_time = datetime.strptime(end, '%Y-%m-%d %H:%M')
urlBase = "http://jsoc.stanford.edu/data/aia/synoptic"
locPath = "/Users/lxy/Desktop/Rice/Su 21/Solar_Weather_Forecast/data/SDOAIA"
count = (end_time - start_time) // delta            # number of data points

def process(events, *argv):
    """

    """
    print(argv)
    filename = 'maps_1024.npz'
    label_file = 'label.npz'
    if "256" in argv:
        filename = 'maps_256.npz'

    src_list = np.zeros((count, 1024, 1024))
    tar_list = np.zeros((count,))
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
        thisFile = "AIA"+ yr + mo + da + "_" + ho + mi + "_0094.fits"

        # Check if the file exists
        if not path.isfile(base + thisFile):
            src_list[data_index] = np.zeros((1024, 1024))
            tar_list[data_index] = 0
            data_index += 1
            continue

        try:
            # lis = sunpy.io.fits.read(base + thisFile)
            smap094 = sunpy.map.Map(base + thisFile)

            # data shape is (1024, 1024)
            # data, header = lis[1]
            data = smap094.data


            if "mask" in argv:
                hpc_coords = all_coordinates_from_map(smap094)
                r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / smap094.rsun_obs

                mask = ma.masked_greater_equal(r, 1)
                data = ma.array(data, mask=mask.mask, fill_value = -999).filled() # fillded with -999
                # print(data)

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

            # Create label, right now it's just 0/1
            # future, it can include different classes and number of events and more
            while True:
                if label_index == len(events):
                    break
                label = events[label_index]
                label_year = "19" + label[0] if int(label[0]) > 21 else "20" + label[0]
                dt = datetime(int(label_year), int(label[1]), int(label[2]), hour = int(label[5][:2]), minute = int(label[5][3:]))
                if time > dt:    
                    label_index += 1
                elif time < dt and time + delta > dt:
                    tar = 1
                    label_index += 1
                else:
                    break

            print(yr + mo + da + "_" + ho + mi)
            print(tar)

            # if "normalize" in argv:
            #     data = normalize(data)

            src_list[data_index] = data
            tar_list[data_index] = tar
            data_index += 1

        except FileNotFoundError:
            pass

    # src_images = asarray(src_list)
    # tar_images = asarray(tar_list)

    print(src_list.shape)
    print(tar_list.shape)

    if "256" in argv:
        src_list = downsample_256(src_list)

    if "saveFile" in argv:
        print('Loaded: ', src_list.shape)
        savez_compressed(filename, src_list, tar_list)
        print('Saved dataset: ', filename)


def downsample_256(src, *argv):
    """Use max-pooling to down-sample

    Args:
        src ([np.array]): array of size N, h, w

    Returns:
        [type]: [description]
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
    # print(events)
    process(events, "256", "mask", "normalize", "show", "saveFile")
    # plot_day("2019", "03", "09")
