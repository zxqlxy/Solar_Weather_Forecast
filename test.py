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

##################### Optimize for max_pool ################### 


from numpy import load
data = load('201102_256.npz')
src_images, tar = data["arr_0"], data["arr_1"]
print(tar)

