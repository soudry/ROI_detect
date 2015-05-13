# -*- coding: utf-8 -*-
"""
Created on Fri May  8 11:15:11 2015

@author: Daniel
"""
# Example Script
from __future__ import division

from numpy.random import randn, randint
from numpy import zeros, transpose, min, max, array, prod, percentile, outer
from time import sleep
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from BlockGroupLasso import gaussian_group_lasso, GetCenters, GetROI, GetActivity
from BlockLocalNMF import LocalNMF, RegionAdd

data_source = 3
plt.close('all')

# Fetch Data
if data_source == 1:  # generate 2D model data
    T = 30  # duration of the simulation
    sz = (150, 100)  # size of image
    sig = (5, 5)  # neurons size
    foo = 0.1 * randn(*((T,) + sz))
    bar = zeros((T,) + sz)
    N = 15  # number of neurons
    lam = 1
    for i in range(N):
        ind = tuple([randint(x) for x in sz])
        for j in range(T):
            bar[(j,) + ind] = abs(randn())
    data = foo + 10 * gaussian_filter(bar, (0,) + sig)
    TargetArea = N * prod(2 * array(sig)) / prod(sz)
    TargetRange = [TargetArea * 0.8, TargetArea * 1.2]
    NonNegative = True
    lam = 1
elif data_source == 2:   # Use experimental 2D data
    mat = loadmat('Datasets/data_exp2D')
    data = transpose(mat['data'], [2, 0, 1])
    sig = (6, 6)  # estimated neurons size
    N = 40  # estimated number of neurons
    TargetArea = N * prod(2 * array(sig)) / prod(data[0, :, :].shape)
    TargetRange = [TargetArea * 0.8, TargetArea * 1.2]
    NonNegative = True
    lam = 1
elif data_source == 3:   # Use experimental 3D data
    mat = loadmat('Datasets/data_exp3D')
    data = transpose(mat['data'], [3, 0, 1, 2])
    sig = (2, 2, 2)  # neurons size
    TargetRange = [0.005, 0.015]
    NonNegative = True
    lam = 0.001


# Run source detection algorithms

#    TargetRange=[]
x = gaussian_group_lasso(
    data, sig, lam, NonNegative=NonNegative, TargetAreaRatio=TargetRange, verbose=True)
#    pic = std(x, 0)
#    z = std(data, 0)
# I think Misha told me once 90% percentile is more robust then max, and
# more sentsitive the std
pic_x = percentile(x, 90, axis=0)
pic_data = percentile(data, 90, axis=0)
# centers extracted from fista output using RegionalMax
cent = GetCenters(pic_x)
# ROI around each center, using watersheding on non-zero regions
ROI = GetROI(pic_x,  (array(cent)[:-1]).T)
# temporal traces of activity for each neuron, averaged over each ROI
activity = GetActivity(x, ROI)
#    residual=array(data.transpose(list(range(1, len(cent))) + [0]))`
residual = array(data)
MSE_array, shapes, activity, boxes = LocalNMF(
    residual, (array(cent)[:-1]).T, activity, sig, NonNegative=NonNegative, verbose=True)

L = len(shapes)  # number of detected neurons
denoised_data = 0 * array(residual)
for ll in range(L):  # add all detected neurons
    denoised_data = RegionAdd(
        denoised_data, outer(activity[ll], shapes[ll],), boxes[ll])
pic_denoised = percentile(denoised_data, 90, axis=0)


# Plot Results
ax = plt.subplot(131)
ax.scatter(cent[1], cent[0], s=4 * sig[1],  marker='o', c='white')
plt.hold(True)
#    ax.scatter(peaks[1],peaks[0],s=2*sig[1],marker='o',c='white')
ax.set_title('Data + centers')
ax.imshow(pic_data if data_source != 3 else pic_data.max(-1))
ax2 = plt.subplot(132)
ax2.scatter(cent[1], cent[0], s=4 * sig[1],  marker='o', c='white')
ax2.imshow(pic_x if data_source != 3 else pic_x.max(-1))
ax2.set_title('Inferred x')
ax3 = plt.subplot(133)
ax3.scatter(cent[1], cent[0], s=4 * sig[1],  marker='o', c='white')
ax3.imshow(pic_denoised if data_source != 3 else pic_denoised.max(-1))
ax3.set_title('Denoised data')

fig = plt.figure()
plt.plot(MSE_array)


# Video Results
dt = 1e-2
fig = plt.figure()
ax = fig.add_subplot(111)
mi = min(data)
ma = max(data)
for ii in range(data.shape[0]):
    sleep(dt)
    ax = plt.subplot(131)
    ax.scatter(cent[1], cent[0], s=4 * sig[1], marker='o', c='white')
    plt.hold(True)
#        ax.scatter(peaks[1],peaks[0],s=3*sig[1],marker='x',c='black')
#        plt.hold(True)
    ax.imshow(data[ii] if data_source != 3 else data[ii].max(-1),
              vmin=mi, vmax=ma, aspect='auto')
    ax.set_title('Data + centers')
    plt.draw()
    plt.hold(False)
    ax2 = plt.subplot(132)
    ax2.scatter(cent[1], cent[0], s=4 * sig[1], marker='o', c='white')
    plt.hold(True)
    ax2.imshow(residual[ii] if data_source != 3 else residual[
               ii].max(-1), vmin=mi, vmax=ma, aspect='auto')
    ax2.set_title('Residual')
    plt.draw()
    plt.hold(False)
    ax3 = plt.subplot(133)
    ax3.scatter(cent[1], cent[0], s=4 * sig[1], marker='o', c='white')
    plt.hold(True)
    ax3.imshow(denoised_data[ii] if data_source != 3 else denoised_data[
               ii].max(-1), vmin=mi, vmax=ma, aspect='auto')
    ax3.set_title('Denoised')
    plt.draw()
    plt.hold(False)
