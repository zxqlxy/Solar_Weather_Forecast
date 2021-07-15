import time
import numpy as np 


##################### Optimize for max_pool ################### 

# Naive method
def downsample_256(src, *argv):
    """
    Use max-pooling to down-sample
    """
    res = []
    for i in range(256):
        line = []
        for j in range(256):
            num = - float("inf")
            for x in range(4):
                for y in range(4):
                    if src[4 * i + x][4 * j + y] > num:
                        num = src[4 * i + x][4 * j + y]
            line.append(num)
        res.append(line)
    return res

# Using numpy, 60x faster
def max_pool(x):
    """Return maximum in groups of 4x4 for a N,h,w image"""
    N,h,w = x.shape
    x = x.reshape(1, 1024, 1024)
    res = np.amax([arr[:, i//4::4,i%4::4] for i in range(16)], axis = 0)
    return res.reshape(256, 256)


# arr = np.array([[i for i in range(1024)] for j in range(1024)])
# arr = arr.reshape(1, 1024, 1024)

# t0 = time.time()
# max_pool(arr)
# print(time.time() - t0)

# arr = arr.reshape(1024, 1024)

# t1 = time.time()
# downsample_256(arr)
# print(time.time() - t1)

##################### Concatenate files ################### 


from numpy import savez_compressed

from numpy import load
# data1 = load('maps_256_3500_non_flares_1.npz')
# data2 = load('maps_256_3500_non_flares_2.npz')
# data1 = load('maps_256_2490_flares.npz')
# data2 = load('maps_256_3500_flares.npz')
data3 = load('maps_256_3500_non_flares_1.npz')

print("finished loading")

### This removes the None sample
# src = data1['arr_0'][data1['arr_0'].sum(axis=(1,2,3)) != 0]
# tar = data1['arr_1'][:2490]
# src2 = data2['arr_0'][data2['arr_0'].sum(axis=(1,2,3)) != 0]
# tar = data1['arr_1'][:3500]
# src3 = data3['arr_0'][data3['arr_0'].sum(axis=(1,2,3)) != 0]
# tar = data3['arr_1'][:818]
# print(src.shape, src2.shape, src3.shape)

# src = np.concatenate((data1['arr_0'], data2['arr_0']), axis=0)
# tar = np.concatenate((data1['arr_1'], data2['arr_1']), axis=0)
# src = np.concatenate((np.concatenate((data1['arr_0'], data2['arr_0']), axis=0), data3['arr_0']), axis=0)
# tar = np.concatenate((np.concatenate((data1['arr_1'], data2['arr_1']), axis=0), data3['arr_1']), axis=0)
    
src = data3['arr_0'][:817]
tar = data3['arr_1'][:817]
print(src.shape)
print(tar.shape)


savez_compressed("maps_256_818_non_flares_test.npz", src, tar)

# import matplotlib.colors
# import matplotlib.pyplot as plt
# from matplotlib.patches import ConnectionPatch

# import astropy.units as u
# from astropy.coordinates import SkyCoord

# import sunpy.map
# # from sunpy.data.sample import HMI_LOS_IMAGE

# magnetogram = sunpy.map.Map("E:\\xl73\\data\\hmi\\hmi_m_45s_2010_05_13_00_01_30_tai_magnetogram.fits").rotate()
# hpc_coords = sunpy.map.all_coordinates_from_map(magnetogram)
# mask = ~sunpy.map.coordinate_is_on_solar_disk(hpc_coords)
# magnetogram_big = sunpy.map.Map(magnetogram.data, magnetogram.meta, mask=mask)
# fig = plt.figure(figsize=(7.2, 4.8))
# norm = matplotlib.colors.SymLogNorm(50, vmin=-7.5e2, vmax=7.5e2)
# ax1 = fig.add_subplot(121, projection=magnetogram_big)
# magnetogram_big.plot(axes=ax1, cmap='RdBu_r', norm=norm, annotate=False,)
# magnetogram_big.draw_grid(axes=ax1, color='black', alpha=0.25, lw=0.5)
# plt.show()
