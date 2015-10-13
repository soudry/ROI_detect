from numpy import min, max, asarray, percentile, zeros, ones, dot, \
    reshape, r_, ix_, arange, exp, nan_to_num, prod, mean, sqrt, repeat
from scipy.signal import welch


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


def LocalNMF(data, centers, sig, NonNegative=True, iters=10, verbose=False,
             adaptBias=True, iters0=[80], mbs=[30], ds=2):
    """
    Parameters
    ----------
    data : array, shape (T, X, Y[, Z])
        block of the data
    centers : array, shape (L, D)
        L centers of suspected neurons where D is spatial dimension (2 or 3)
    sig : array, shape (D,)
        size of the gaussian kernel in different spatial directions
    NonNegative : boolean
        if True, neurons should be considered as non-negative
    iters : int
        number of final iterations on whole data
    verbose : boolean
        print progress and record MSE if true (about 2x slower)
    adaptBias : boolean
        subtract rank 1 estimate of bias
    iters0 : list
        numbers of initial iterations on subset
    mbs : list
        minibatchsizes for temporal downsampling
    ds : int
        factor for spatial downsampling


    Returns
    -------
    MSE_array : list (empty if verbose is False)
        Mean square error during algorithm operation
    shapes : array, shape (L+adaptBias, X, Y (,Z))
        the neuronal shape vectors
    activity : array, shape (L+adaptBias, T)
        the neuronal activity for each shape
    boxes : array, shape (L, D, 2)
        edges of the boxes in which each neuronal shapes lie
    """

    # Initialize Parameters
    dims = data.shape
    D = len(dims)
    R = 3 * asarray(sig)  # size of bounding box is 3 times size of neuron
    L = len(centers)
    shapes = []
    mask = []
    boxes = zeros((L, D - 1, 2), dtype=int)
    MSE_array = []
    mb = mbs[0] if iters0[0] > 0 else 1
    activity = zeros((L, dims[0] / mb))
    if iters0[0] == 0:
        ds = 1

### Function definitions ###
    # Estimate noise level
    def GetSnPSD(Y):
        L = len(Y)
        ff, psd_Y = welch(Y, nperseg=round(L / 8))
        sn = sqrt(mean(psd_Y[ff > .3] / 2))
        return sn
    noise = zeros(L)

    def HALS(data, S, activity, skip=[], check_skip=0, iters=1):
        idx = asarray(filter(lambda x: x not in skip, range(len(activity))))
        A = S[idx].dot(data.T)
        B = S[idx].dot(S.T)
        for ii in range(iters):
            for k, ll in enumerate(idx):
                if check_skip and ii == iters - 1:
                    a0 = activity[ll].copy()
                activity[ll] += nan_to_num((A[k] - np.dot(B[k], activity)) / B[k, ll])
                if NonNegative:
                    activity[ll][activity[ll] < 0] = 0
            # skip neurons whose shapes already converged
                if check_skip and ll < L and ii == iters - 1:
                    if check_skip == 1:  # compute noise level only once
                        noise[ll] = GetSnPSD(a0) / a0.mean()
                    if np.allclose(a0, activity[ll] / activity[ll].mean(), 1e-4, noise[ll]):
                        skip += [ll]
        C = activity[idx].dot(data)
        D = activity[idx].dot(activity.T)
        for _ in range(iters):
            for k, ll in enumerate(idx):
                if ll == L:
                    S[ll] += nan_to_num((C[k] - np.dot(D[k], S)) / D[k, ll])
                else:
                    S[ll, mask[ll]] += nan_to_num((C[k, mask[ll]]
                                                   - np.dot(D[k], S[:, mask[ll]])) / D[k, ll])
                if NonNegative:
                    S[ll][S[ll] < 0] = 0
        return S, activity, skip

    def HALS4activity(data, S, activity, iters=1):
        A = S.dot(data.T)
        B = S.dot(S.T)
        for _ in range(iters):
            for ll in range(L + adaptBias):
                activity[ll] += nan_to_num((A[ll] - np.dot(B[ll].T, activity)) / B[ll, ll])
                if NonNegative:
                    activity[ll][activity[ll] < 0] = 0
        return activity

    def HALS4shape(data, S, activity, iters=1):
        C = activity.dot(data)
        D = activity.dot(activity.T)
        for _ in range(iters):
            for ll in range(L + adaptBias):
                if ll == L:
                    S[ll] += nan_to_num((C[ll] - np.dot(D[ll], S)) / D[ll, ll])
                else:
                    S[ll, mask[ll]] += nan_to_num((C[ll, mask[ll]]
                                                   - np.dot(D[ll], S[:, mask[ll]])) / D[ll, ll])
                if NonNegative:
                    S[ll][S[ll] < 0] = 0
        return S


### Initialize shapes, activity, and residual ###
    data0 = data[:len(data) / mb * mb].reshape((-1, mb) + data.shape[1:]).mean(1)
    if D == 4:
        data0 = data0.reshape(
            len(data0), dims[1] / ds, ds, dims[2] / ds, ds, dims[3] / ds, ds)\
            .mean(2).mean(3).mean(4)
        activity = data0[:, map(int, centers[:, 0] / ds), map(int, centers[:, 1] / ds),
                         map(int, centers[:, 2] / ds)].T
    else:
        data0 = data0.reshape(len(data0), dims[1] / ds, ds, dims[2] / ds, ds).mean(2).mean(3)
        activity = data0[:, map(int, centers[:, 0] / ds), map(int, centers[:, 1] / ds)].T
    # for i,d in enumerate(dims[1:]):
    #     data0 = data0.reshape(data0.shape[:1+i] + (d / ds, ds, -1)).mean(2+i)
    dims0 = data0.shape

    data0 = data0.reshape(dims0[0], -1)
    data = data.astype('float').reshape(dims[0], -1)
    # float is faster than float32, presumable float32 gets converted later on
    # to float again and again
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
        activity = np.r_[activity, ones((1, dims0[0]))]

### Get shape estimates on subset of data ###
    if iters0[0] > 0:
        skip = []
        for it in range(len(iters0)):
            for kk in range(iters0[it]):
                activity = HALS4activity(data0, S, activity)
                S = HALS4shape(data0, S, activity)
            if it < len(iters0) - 1:
                mb = mbs[it + 1]
                data0 = data[:len(data) / mb * mb].reshape(-1, mb, prod(dims[1:])).mean(1)
                data0 = data0.reshape(len(data0), dims[1] /
                                      ds, ds, dims[2] / ds, ds).mean(-1).mean(-2)
                data0.shape = (len(data0), -1)
                activity = ones((L + adaptBias, len(data0))) * activity.mean(1).reshape(-1, 1)
                activity = HALS4activity(data0, S, activity)

    ### Back to full data ##
        activity = ones((L + adaptBias, dims[0])) * activity.mean(1).reshape(-1, 1)
        if D==4:
            S = repeat(repeat(repeat(S.reshape((-1,) + dims0[1:]), ds, 1), ds, 2), ds, 3).reshape(L + adaptBias, -1)
        else:
            S = repeat(repeat(S.reshape((-1,) + dims0[1:]), ds, 1), ds, 2).reshape(L + adaptBias, -1)
        for ll in range(L):
            boxes[ll] = GetBox(centers[ll], R, dims[1:])
            temp = zeros(dims[1:])
            temp[map(lambda a: slice(*a), boxes[ll])] = 1
            mask[ll] = np.where(temp.ravel())[0]

        activity = HALS4activity(data, S, activity, 7)
        S = HALS4shape(data, S, activity, 7)

#### Main Loop ####
    skip = []
    for kk in range(iters):
        S, activity, skip = HALS(data, S, activity, skip, iters=10)  # , check_skip=kk + 1)
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
        if verbose:
            residual = data - activity.T.dot(S)
            MSE = dot(residual.ravel(), residual.ravel()) / data.size
            print('{0:1d}: MSE = {1:.3f}'.format(kk, MSE))
            if kk == (iters - 1):
                print('Maximum iteration limit reached')
            MSE_array.append(MSE)

    return asarray(MSE_array), S.reshape((-1,) + dims[1:]), activity, boxes


# example
import numpy as np

T = 50
X = 200
Y = 100
data = np.random.randn(T, X, Y)
centers = asarray([[40, 30]])
data[:, 40, 30] += np.random.randn(T)
sig = [3, 3]
R = 3 * np.array(sig)
dims = data.shape
MSE_array, shapes, activity, boxes = LocalNMF(
    data, centers, sig, NonNegative=True, verbose=True)
