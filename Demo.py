# Example Script
from __future__ import division

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.random import randn, randint
from numpy import zeros, transpose, min, max, array, prod, percentile, outer
from scipy.io import loadmat
from scipy.ndimage.filters import gaussian_filter
from sys import argv
from BlockGroupLasso import gaussian_group_lasso, GetCenters, GetROI, GetActivity
from BlockLocalNMF import LocalNMF, RegionAdd

data_source = 1 if len(argv) == 1 else int(argv[1])
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
x = gaussian_group_lasso(data, sig, lam, NonNegative=NonNegative,
                         TargetAreaRatio=TargetRange, verbose=True, adaptBias=True)
pic_x = percentile(x, 95, axis=0)
pic_data = percentile(data, 95, axis=0)
# centers extracted from fista output using RegionalMax
cent = GetCenters(pic_x)
# ROI around each center, using watersheding on non-zero regions
ROI = GetROI(pic_x,  (array(cent)[:-1]).T)
# temporal traces of activity for each neuron, averaged over each ROI
activity = GetActivity(x, ROI)

MSE_array, shapes, activity, boxes, background = LocalNMF(
    data, (array(cent)[:-1]).T, activity, sig,
    NonNegative=NonNegative, verbose=True, adaptBias=True)

L = len(shapes)  # number of detected neurons
denoised_data = 0 * data
for ll in range(L):  # add all detected neurons
    denoised_data = RegionAdd(
        denoised_data, outer(activity[ll], shapes[ll],), boxes[ll])
pic_denoised = percentile(denoised_data, 95, axis=0)
residual = data - denoised_data - background

# Plot Results
plt.figure(figsize=(12, 4. * data.shape[1] / data.shape[2]))
ax = plt.subplot(131)
ax.scatter(cent[1], cent[0], s=7 * sig[1],  marker='o', c='white')
plt.hold(True)
ax.set_title('Data + centers')
ax.imshow(pic_data if data_source != 3 else pic_data.max(-1))
ax2 = plt.subplot(132)
ax2.scatter(cent[1], cent[0], s=7 * sig[1],  marker='o', c='white')
ax2.imshow(pic_x if data_source != 3 else pic_x.max(-1))
ax2.set_title('Inferred x')
ax3 = plt.subplot(133)
ax3.scatter(cent[1], cent[0], s=7 * sig[1],  marker='o', c='white')
ax3.imshow(pic_denoised if data_source != 3 else pic_denoised.max(-1))
ax3.set_title('Denoised data')
plt.show()

fig = plt.figure()
plt.plot(MSE_array)
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.show()

# Video Results
fig = plt.figure(figsize=(12, 4. * data.shape[1] / data.shape[2]))
mi = min(data)
ma = max(data)
ii = 0
ax = plt.subplot(131)
ax.scatter(cent[1], cent[0], s=7 * sig[1], marker='o', c='white')
im = ax.imshow(data[ii] if data_source != 3 else data[ii].max(-1), vmin=mi, vmax=ma)
ax.set_title('Data + centers')
ax2 = plt.subplot(132)
ax2.scatter(cent[1], cent[0], s=7 * sig[1], marker='o', c='white')
im2 = ax2.imshow(residual[ii] if data_source != 3 else residual[ii].max(-1), vmin=mi, vmax=ma)
ax2.set_title('Residual')
ax3 = plt.subplot(133)
ax3.scatter(cent[1], cent[0], s=7 * sig[1], marker='o', c='white')
im3 = ax3.imshow(denoised_data[ii] if data_source !=
                 3 else denoised_data[ii].max(-1), vmin=mi, vmax=ma)
ax3.set_title('Denoised')
def update(ii):
    im.set_data(data[ii] if data_source != 3 else data[ii].max(-1))
    im2.set_data(residual[ii] if data_source != 3 else residual[ii].max(-1))
    im3.set_data(denoised_data[ii] if data_source != 3 else denoised_data[ii].max(-1))
ani = animation.FuncAnimation(fig, update, frames=len(data), blit=False, interval=30,
                              repeat=False)
plt.show()
