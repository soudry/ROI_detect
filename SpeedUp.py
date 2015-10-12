from numpy import min, max, array, asarray, percentile, zeros, ones, dot, \
    reshape, r_, ix_, arange, exp, nan_to_num, argsort, prod, mean, sqrt, repeat
from time import time
import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt


def GetBox(centers, R, dims):
    D = len(R)
    box = zeros((D, 2), dtype=int)
    for dd in range(D):
        box[dd, 0] = max((centers[dd] - R[dd], 0))
        box[dd, 1] = min((centers[dd] + R[dd] + 1, dims[dd]))
    return box


def RegionAdd(Z, X, box):
    # Parameters
    #  Z : array, shape (T, X, Y[, Z]), dataset
    #  box : array, shape (D, 2), array defining spatial box to put X in
    #  X : array, shape (T, prod(diff(box,1))), Input
    # Returns
    #  Z : array, shape (T, X, Y[, Z]), Z+X on box region
    Z[[slice(len(Z))] + list(map(lambda a: slice(*a), box))
      ] += reshape(X, (r_[-1, box[:, 1] - box[:, 0]]))
    return Z


def RegionCut(X, box):
    # Parameters
    #  X : array, shape (T, X, Y[, Z])
    #  box : array, shape (D, 2), region to cut
    # Returns
    #  res : array, shape (T, prod(diff(box,1))),
    dims = X.shape
    return X[[slice(dims[0])] + list(map(lambda a: slice(*a), box))].reshape((dims[0], -1))


def GetCenters(image):
    """ Take a image and detect the peaks using local maximum filter.
    input: a 2D image
    output: peaks list, in which
        peaks[0] - y coordinates
        peaks[1] - x coordinates
        peaks[2] - magnitude ("height") of peak
    """
    from skimage.feature import peak_local_max
    peaks = peak_local_max(image, min_distance=3, threshold_rel=.03, exclude_border=False).T
    magnitude = image[list(peaks)]
    indices = argsort(magnitude)[::-1]
    peaks = list(peaks[:, indices]) + [magnitude[indices]]
    return peaks


def LocalNMF(data, centers, sig, NonNegative=True,
             tol=1e-5, iters=10, verbose=False, adaptBias=True, iters0=[0], mbs=None):
    """
    Parameters
    ----------
    data : array, shape (T, X, Y[, Z])
        block of the data
    centers : array, shape (L, D)
        L centers of suspected neurons where D is spatial dimension (2 or 3)
    activity : array, shape (L, T)
        traces of temporal activity
    sig : array, shape (D,)
        size of the gaussian kernel in different spatial directions
    NonNegative : boolean
        if True, neurons should be considered as non-negative
    tol : float
        tolerance for stopping algorithm
    iters : int
        maximum number of iterations
    verbose : boolean
        print progress if true
    adaptBias : boolean
        subtract rank 1 estimate of bias

    Returns
    -------
    MSE_array : list
        Mean square error during algorithm operation
    shapes : list (length L) of lists (var length)
        the neuronal shape vectors
    activity : array, shape (L, T)
        the neuronal activity for each shape
    boxes : array, shape (L, D, 2)
        edges of the boxes in which each neuronal shapes lie
    """
    t = time()

    # Initialize Parameters
    dims = data.shape
    D = len(dims)
    R = 3 * array(sig)  # size of bounding box is 3 times size of neuron
    L = len(centers)
    shapes = []
    mask = []
    boxes = zeros((L, D - 1, 2), dtype=int)
    MSE_array = []
    mb = mbs[0] if iters0[0] > 0 else 1
    activity = zeros((L, dims[0] / mb))
    ds = 2 if iters0[0] > 0 else 1  # downscale
    tls = []
    tsub = 0


### Function definitions ###
    # Estimate noise level
    def GetSnPSD(Y):
        L = len(Y)
        ff, psd_Y = welch(Y, nperseg=round(L / 8))
        sn = sqrt(mean(psd_Y[ff > .3] / 2))
        return sn
    noise = zeros(L)

    def HALS(data, S, activity, skip=[], check_skip=0):
        A = data.dot(S.T)
        B = S.dot(S.T)
        for ll in range(L + adaptBias):
            if ll in skip:
                continue
            if check_skip:
                a0 = activity[ll].copy()
            activity[ll] += nan_to_num((A[:, ll] - np.dot(activity.T, B[:, ll])) / B[ll, ll])
            if NonNegative:
                activity[ll][activity[ll] < 0] = 0
        # skip neurons whose shapes already converged
            if check_skip and ll < L:
                if check_skip == 1:  # compute noise level only once
                    noise[ll] = GetSnPSD(a0)
                if np.allclose(a0, activity[ll] / activity[ll].mean(), 1e-4, .01 * noise[ll]):
                    skip += [ll]

        C = activity.dot(data)
        D = activity.dot(activity.T)
        for ll in range(L + adaptBias):
            if ll in skip:
                continue
            if ll == L:
                S[ll] += nan_to_num((C[ll] - np.dot(D[ll], S)) / D[ll, ll])
            else:
                S[ll, mask[ll]] += nan_to_num((C[ll, mask[ll]]
                                               - np.dot(D[ll], S[:, mask[ll]])) / D[ll, ll])
            if NonNegative:
                S[ll][S[ll] < 0] = 0

        tsub = time()
        residual = data - activity.T.dot(S)
        tsub -= time()

        return residual, S, activity, skip, tsub

    def HALS4activity(data, S, activity):
        A = S.dot(data.T)
        B = S.dot(S.T)
        for ll in range(L + adaptBias):
            activity[ll] += nan_to_num((A[ll] - np.dot(B[ll].T, activity)) / B[ll, ll])
            if NonNegative:
                activity[ll][activity[ll] < 0] = 0
        tsub = time()
        residual = data - activity.T.dot(S)
        tsub -= time()
        return residual, activity, tsub

    def HALS4shape(data, S, activity):
        C = activity.dot(data)
        D = activity.dot(activity.T)
        for ll in range(L + adaptBias):
            if ll == L:
                S[ll] += nan_to_num((C[ll] - np.dot(D[ll], S)) / D[ll, ll])
            else:
                S[ll, mask[ll]] += nan_to_num((C[ll, mask[ll]]
                                               - np.dot(D[ll], S[:, mask[ll]])) / D[ll, ll])
            if NonNegative:
                S[ll][S[ll] < 0] = 0
        tsub = time()
        residual = data - activity.T.dot(S)
        tsub -= time()
        return residual, S, tsub


### Initialize shapes, activity, and residual ###
    print 'init', time() - t
    # from skimage.transform import downscale_local_mean
    data0 = data.reshape((-1, mb) + data.shape[-2:]).mean(1)
    # data0 = asarray([downscale_local_mean(r, (ds, ds)) for r in data0])
    data0 = data0.reshape(len(data0), dims[1] / ds, ds, dims[2] / ds, ds).mean(-1).mean(-2)
    dims0 = data0.shape
    activity = data0[:, map(int, centers[:, 0] / ds), map(int, centers[:, 1] / ds)].T
    activity = np.r_[activity, ones((1, dims0[0]))]
    data0 = data0.reshape(dims0[0], -1)
    data = data.astype('float').reshape(dims[0], -1)
    # float is faster than float32, presumable float32 gets converted later on
    # to float agina and again
    for ll in range(L):
        boxes[ll] = GetBox(centers[ll] / ds, R / ds, dims0[1:])
        temp = zeros(dims0[1:])
        temp[map(lambda a: slice(*a), boxes[ll])]=1
        mask += np.where(temp.ravel())
        temp = [(arange(dims[i + 1] / ds) - centers[ll][i] / ds) ** 2 / (2 * (sig[i] / ds) ** 2)
                for i in range(D - 1)]
        temp = exp(-sum(ix_(*temp)))
        temp.shape = (1,) + dims0[1:]
        temp = RegionCut(temp, boxes[ll])
        shapes.append(temp[0])
    S = zeros((L + adaptBias, prod(dims0[1:])))
    for ll in range(L):
        S[ll] = RegionAdd(
            zeros((1,) + dims0[1:]), shapes[ll].reshape(1, -1), boxes[ll]).ravel()
    if adaptBias:
        # Initialize background as 20% percentile
        S[-1] = percentile(data0, 20, 0)

### Get shape estimates on subset of data ###
    if iters0[0] > 0:
        skip = []
        for it in range(len(iters0)):
            for kk in range(iters0[it]):
                print 'subset', time() - t, kk, skip
                _, S, activity, skip, dt = HALS(data0, S, activity, skip)
                tsub += dt
            if it < len(iters0) - 1:
                mb = mbs[it + 1]
                data0 = data.reshape(-1, mb, prod(dims[1:])).mean(1)
                activity = ones((L + adaptBias, len(data0))) * activity.mean(1).reshape(-1, 1)
                residual, activity, dt = HALS4activity(data0, S, activity)
                tsub += dt

    ### Back to full data ##
        activity = ones((L + adaptBias, dims[0])) * activity.mean(1).reshape(-1, 1)
        S = repeat(repeat(S.reshape((-1,) + dims0[1:]), 2, 1), 2, 2).reshape(L + adaptBias, -1)
        for ll in range(L):
            boxes[ll] = GetBox(centers[ll], R, dims[1:])
            temp = zeros(dims[1:])
            temp[map(lambda a: slice(*a), boxes[ll])] = 1
            mask[ll] = np.where(temp.ravel())[0]

        for _ in range(1):
            # replace data0 by residual for HALS using residual
            res, activity, dt = HALS4activity(data, S, activity)
            tsub += dt


#### Main Loop ####
    skip = []
    for kk in range(iters):
        print 'main', time() - t, kk, tsub
        residual, S, activity, skip, dt = HALS(data, S, activity, skip)  # , kk + 1)
        # Recenter
        # if kk % 30 == 20:
        #     for ll in range(L):
        #         shp = shapes[ll].reshape(np.ravel(np.diff(boxes[ll])))
        #         com = boxes[ll][:, 0] + round(center_of_mass(shp))
        #         newbox = GetBox(com, R, dims[1:])
        #         if any(newbox != boxes[ll]):
        #             newshape = zeros(np.ravel(np.diff(newbox)))
        #             lower = vstack([newbox[:, 0], boxes[ll][:, 0]]).max(0)
        #             upper = vstack([newbox[:, 1], boxes[ll][:, 1]]).min(0)
        #             newshape[lower[0] - newbox[0, 0]:upper[0] - newbox[0, 0],
        #                      lower[1] - newbox[1, 0]:upper[1] - newbox[1, 0]] =
        #                 shp[lower[0] - boxes[ll][0, 0]:upper[0] - boxes[ll][0, 0],
        #                     lower[1] - boxes[ll][1, 0]:upper[1] - boxes[ll][1, 0]]
        #             shapes[ll] = newshape.reshape(-1)
        #             boxes[ll] = newbox

        # Measure MSE
        MSE = dot(residual.ravel(), residual.ravel())
        tsub += dt
        tls += [[time() - t + tsub, MSE]]
        if verbose:
            print('{0:1d}: MSE = {1:.3f}'.format(kk, MSE))
        if kk > 0 and abs(1 - MSE / MSE_array[-1]) < tol:
            break
        if kk == (iters - 1):
            print('Maximum iteration limit reached')
        MSE_array.append(MSE)
    if adaptBias:
        return tls, S, activity, boxes  # , outer(b_t, b_s).reshape(dims)
    else:
        return MSE_array, S, activity, boxes

########################################################

if __name__ == "__main__":
    # Fetch Data, take only 100x100 patch to not have to wait minutes
    sig = (4, 4)
    lam = 40
    data = np.asarray([np.load('../zebrafish/ROI_zebrafish/data/1/nparrays/TM0%04d_200-400_350-550_15.npy' % t)
                       for t in range(3000)])[:, : 100, : 100]
    x = np.load('x.npy')[:, : 100, : 100]  # x is stored result from grouplasso
    pic_x = np.percentile(x, 95, 0)
    cent = GetCenters(pic_x)

    # iterls = np.outer([10, 20, 40, 60, 80], np.ones(2, dtype=int)) / 2
    # iterls = [10, 20, 40, 60, 80]
    iterls = [80]
    MSE_array = [LocalNMF(data, (array(cent)[:-1]).T, sig, verbose=True, iters=20, iters0=[i], mbs=[30])[0]
                 for i in iterls]
    plt.figure()
    for i, m in enumerate(MSE_array):
        plt.plot(np.array(m)[:, 0], np.array(m)[:, 1] / data.size, label=iterls[i])
    plt.legend(title='subset iterations')
    plt.xlabel('Walltime')
    plt.ylabel('MSE')
    plt.show()

    # tls, shapes, activity, boxes, background = LocalNMF(
    #     data, (array(cent)[:-1]).T, sig, verbose=True, iters=1, iters0=[60], mbs=[30])
    # for ll in range(len(shapes)):
    # figure()
    # imshow(shapes[ll].reshape(np.diff(boxes[ll], 1).ravel()))
    # figure()
    # imshow(shapes[-1].reshape(np.diff(boxes[-1], 1).ravel()))
