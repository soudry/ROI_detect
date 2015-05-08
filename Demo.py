# -*- coding: utf-8 -*-
"""
Created on Fri May  8 11:15:11 2015

@author: Daniel
"""
# Example Script

if __name__ == "__main__":
    from numpy.random import randn, randint
    from numpy import zeros, transpose, std, min, max, array
    from time import sleep
    from scipy.io import loadmat
    import matplotlib.pyplot as plt
    from scipy.ndimage.filters import gaussian_filter
    from BlockGroupLasso import gaussian_group_lasso, GetCenters, GetROI, GetActivity
    from BlockLocalNMF import LocalNMF

    data_source = 1

    if data_source == 1:  # generate 2D model data
        T = 30
        sz = (50, 100)
        sig = (5, 5)  # neurons size
        foo = 0.1 * randn(*((T,) + sz))
        bar = zeros((T,) + sz)
        for i in range(20):
            ind = tuple([randint(x) for x in sz])
            for j in range(T):
                bar[(j,) + ind] = randn()
        data = foo + 10 * gaussian_filter(bar, (0,) + sig)
    elif data_source == 2:   # Use experimental 2D data
        mat = loadmat('Datasets/data_exp2D')
        data = transpose(mat['data'], [2, 0, 1])
        sig = (5, 5)  # neurons size
    elif data_source == 3:   # Use experimental 3D data
        mat = loadmat('Datasets/data_exp3D')
        data = transpose(mat['data'], [3, 0, 1, 2])
        sig = (5, 5, 5)  # neurons size

    TargetRange = [0.03, 0.04]
    lam = 1
#    TargetRange=[]
    x = gaussian_group_lasso(
        data, sig, lam, NonNegative=True, TargetAreaRatio=TargetRange, verbose=True)
    pic = std(x, 0)
    cent = GetCenters(pic)
    ROI = GetROI(pic,  (array(cent)[:2]).T)
    activity = GetActivity(x, ROI)
    MSE_array, shapes, activity, boxes = LocalNMF(data.transpose(1, 2, 0),
                                                  (array(cent)[:2]).T, activity, sig, NonNegative=True, verbose=True)

    z = std(data, 0)


# Plot FISTA

    ax = plt.subplot2grid((1, 2), (0, 0), colspan=1)
    ax.imshow(z)
    plt.hold(True)
    ax.scatter(cent[1], cent[0], s=2 * sig[1], marker='x', c='black')
#    ax.scatter(peaks[1],peaks[0],s=2*sig[1],marker='o',c='white')
    ax.set_title('Data with detected centers')
    ax2 = plt.subplot2grid((1, 2), (0, 1), colspan=1)
    ax2.imshow(pic)
    ax2.set_title('Inferred x')

    # Video
    dt = 1e-2
    fig = plt.figure()
    ax = fig.add_subplot(111)
    mi = min(data)
    ma = max(data)
    for ii in range(data.shape[0]):
        sleep(dt)
        ax.set_title('Data with detected centers')
        ax.scatter(cent[1], cent[0], s=2 * sig[1], marker='o', c='white')
        plt.hold(True)
#        ax.scatter(peaks[1],peaks[0],s=3*sig[1],marker='x',c='black')
#        plt.hold(True)
        ax.imshow(data[ii, :, :], vmin=mi, vmax=ma, aspect='auto')
        plt.draw()
        plt.hold(False)


# Plot NMF

    # plot()
