from numpy import min, max, asarray, percentile, zeros, exp, unravel_index,\
    ones, dot, where, round, reshape, r_, ix_, arange, nan_to_num, argmax,\
    prod, mean, sqrt, repeat, allclose, any, outer, unique, hstack, isnan
from numpy.linalg import norm
from numpy.random import randint
from scipy.signal import welch
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage import median_filter


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
             adaptBias=True, iters0=[80], mbs=[30], ds=None, optimizeCenters=True, thresh=None):
    """
    Parameters
    ----------
    data : array, shape (T, X, Y[, Z])
        block of the data
    centers : array, shape (L, D) or int
        if array : L centers of suspected neurons where D is spatial dimension (2 or 3)
        if int : initial number of randomly placed tiles, ~3-10 times number of neurons
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
    ds : array, shape (D,)
        factor for spatial downsampling in different spatial directions
    optimizeCenters : boolean
        If true, update centers to be center of mass for each source
    thresh : float
        threshold for merging neurons; merge occurs if MSE between
        original 2 components and merged one is below threshold
        if None, no merging is done


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

    # thresh=1 seems good for sigma=(4,4), 5 for (3.,3.)
    # Initialize Parameters
    dims = data.shape
    if isinstance(centers, (int, long)):
        mode = 'rand'
        centers = asarray([randint(0, d, centers) for d in dims[1:]]).T
        centers = centers[np.argsort([data[:, c[0], c[1]].max() for c in centers])[::-1]]
    else:
        mode = 'ctrs'
    D = len(dims)
    R = (3 * asarray(sig)).astype('uint8')  # size of bounding box is 3 times size of neuron
    L = len(centers)
    shapes = []
    mask = []
    boxes = zeros((L, D - 1, 2), dtype=int)
    MSE_array = []
    mb = mbs[0] if iters0[0] > 0 else 1
    activity = zeros((L, dims[0] / mb))
    if iters0[0] == 0 or ds is None:
        ds = 2 * ones(D - 1, dtype='uint8')
    else:
        ds = asarray(ds, dtype='uint8')

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
                activity[ll] += nan_to_num((A[k] - dot(B[k], activity)) / B[k, ll])
                if NonNegative:
                    activity[ll][activity[ll] < 0] = 0
            # skip neurons whose shapes already converged
                if check_skip and ll < L and ii == iters - 1:
                    if check_skip == 1:  # compute noise level only once
                        noise[ll] = GetSnPSD(a0) / a0.mean()
                    if allclose(a0, activity[ll] / activity[ll].mean(), 1e-4, noise[ll]):
                        skip += [ll]
        C = activity[idx].dot(data)
        D = activity[idx].dot(activity.T)
        for _ in range(iters):
            for k, ll in enumerate(idx):
                if ll == L:
                    S[ll] += nan_to_num((C[k] - dot(D[k], S)) / D[k, ll])
                else:
                    S[ll, mask[ll]] += nan_to_num((C[k, mask[ll]]
                                                   - dot(D[k], S[:, mask[ll]])) / D[k, ll])
                if NonNegative:
                    S[ll][S[ll] < 0] = 0
        return S, activity, skip

    def HALS4activity(data, S, activity, iters=1):
        A = S.dot(data.T)
        B = S.dot(S.T)
        for _ in range(iters):
            for ll in range(L + adaptBias):
                activity[ll] += nan_to_num((A[ll] - dot(B[ll].T, activity)) / B[ll, ll])
                if NonNegative:
                    activity[ll][activity[ll] < 0] = 0
        return activity

    def HALS4shape(data, S, activity, iters=1):
        C = activity.dot(data)
        D = activity.dot(activity.T)
        for _ in range(iters):
            for ll in range(L + adaptBias):
                if ll == L:
                    S[ll] += nan_to_num((C[ll] - dot(D[ll], S)) / D[ll, ll])
                else:
                    S[ll, mask[ll]] += nan_to_num((C[ll, mask[ll]]
                                                   - dot(D[ll], S[:, mask[ll]])) / D[ll, ll])
                if NonNegative:
                    S[ll][S[ll] < 0] = 0
        return S

    def recenter(S, boxes, mask, ds):
        dim = dims[1:] / ds
        for ll in range(len(boxes)):
            # com = center_of_mass(S[ll].reshape(dim))
            com = unravel_index(argmax(median_filter(S[ll].reshape(dim), 3)), dim)
            if isnan(com[0]):
                continue
            newbox = GetBox(round(com), R / ds, dim)
            if any(newbox != boxes[ll]):
                temp = zeros(dim)
                temp[map(lambda a: slice(*a), newbox)] = 1
                mask[ll] = where(temp.ravel())[0]
                S[ll] *= temp.ravel()
                boxes[ll] = newbox
        return S, boxes, mask

    def mergeAll(S, activity, boxes, mask, L, ds):
        dim = dims[1:] / ds

        def merge(S, activity, boxes, mask, i, j, th, purge):
            # determine merged component
            sCombined = (S[i] / norm(S[i]) + S[j] / norm(S[j]))
            aCombined = ((activity[i] * norm(S[i]) + activity[j] * norm(S[j])) / 2.)
            sa = outer(activity[i], S[i]) + outer(activity[j], S[j])
            for _ in range(3):
                A = sCombined.dot(sa.T)
                B = sCombined.dot(sCombined)
                aCombined = nan_to_num(A / B)
                if NonNegative:
                    aCombined[aCombined < 0] = 0
                C = aCombined.dot(sa)
                D = aCombined.dot(aCombined)
                sCombined = nan_to_num(C / D)
                if NonNegative:
                    sCombined[sCombined < 0] = 0
            shp = sCombined.reshape(dim)
            com = center_of_mass(shp)
            newbox = GetBox(round(com), R / ds, dim)
            temp = zeros(dim)
            temp[map(lambda a: slice(*a), newbox)] = 1
            newmask = where(temp.ravel())[0]
        # calc MSE
            qq = 0
            for k in newmask:
                tmp = aCombined * sCombined[k] - sa[:, k]
                qq += tmp.dot(tmp)
            for k in filter(lambda a: a not in newmask, unique(hstack([mask[i], mask[j]]))):
                qq += sa[:, k].dot(sa[:, k])
        # merge only if MSE is smaller than some threshold
            if qq < th * len(newmask) * len(aCombined):  # * sqrt(sa.mean()):
                S[i] = sCombined * temp.ravel()
                boxes[i] = newbox
                mask[i] = newmask
                activity[i] = aCombined
                purge += [j]
                if verbose:
                    print 'merged', i, 'and ', j
            return S, activity, boxes, mask, purge
        purge = []
        com = zeros((L, D - 1))
        for ll in range(L):
            com[ll] = center_of_mass(S[ll].reshape(dim))
            if isnan(com[ll, 0]):
                purge += [ll]
        # com = boxes.mean(2)
        for l in range(L - 1):
            if l in purge:
                continue
            for k in range(l + 1, L):
                if k not in purge and norm((com[l] - com[k]) / asarray(sig / ds)) < 2:
                    S, activity, boxes, mask, purge = merge(
                        S, activity, boxes, mask, l, k, thresh, purge)
        idx = filter(lambda x: x not in purge, range(L))
        mask = asarray(mask)[idx]
        boxes = asarray(boxes)[idx]
        if adaptBias:
            idx = asarray(idx + [L])
        S = S[idx]
        activity = activity[idx]
        L = len(mask)
        skip = []
        return S, activity, boxes, mask, L

### Initialize shapes, activity, and residual ###
    data0 = data[:len(data) / mb * mb].reshape((-1, mb) + data.shape[1:]).mean(1).astype('float32')
    if D == 4:
        data0 = data0.reshape(
            len(data0), dims[1] / ds[0], ds[0], dims[2] / ds[1], ds[1], dims[3] / ds[2], ds[2])\
            .mean(2).mean(3).mean(4)
        activity = data0[:, map(int, centers[:, 0] / ds[0]), map(int, centers[:, 1] / ds[1]),
                         map(int, centers[:, 2] / ds[2])].T
    else:
        data0 = data0.reshape(len(data0), dims[1] / ds[0],
                              ds[0], dims[2] / ds[1], ds[1]).mean(2).mean(3)
        activity = data0[:, map(int, centers[:, 0] / ds[0]), map(int, centers[:, 1] / ds[1])].T
    dims0 = data0.shape
    data0 = data0.reshape(dims0[0], -1)
    data = data.astype('float32').reshape(dims[0], -1)
    for ll in range(L):
        boxes[ll] = GetBox(centers[ll] / ds, R / ds, dims0[1:])
        temp = zeros(dims0[1:])
        temp[map(lambda a: slice(*a), boxes[ll])]=1
        mask += where(temp.ravel())
        temp = [(arange(dims[i + 1] / ds[i]) - centers[ll][i] / float(ds[i])) ** 2 / (2 * (sig[i] / float(ds[i])) ** 2)
                for i in range(D - 1)]
        temp = exp(-sum(ix_(*temp)))
        temp.shape = (1,) + dims0[1:]
        temp = RegionCut(temp, boxes[ll])
        shapes.append(temp[0])
    S = zeros((L + adaptBias, prod(dims0[1:])), dtype='float32')
    for ll in range(L):
        S[ll] = RegionAdd(
            zeros((1,) + dims0[1:]), shapes[ll].reshape(1, -1), boxes[ll]).ravel()
    if adaptBias:
        # Initialize background as 20% percentile
        S[-1] = percentile(data0, 20, 0)
        activity = r_[activity, ones((1, dims0[0]), dtype='float32')]

### Get shape estimates on subset of data ###
    if iters0[0] > 0:
        skip = []
        for it in range(len(iters0)):
            for kk in range(iters0[it]):
                S = HALS4shape(data0, S, activity)
                activity = HALS4activity(data0, S, activity)
                if kk > 10:
                    if mode == 'rand':
                        if kk % 3 == 0 and optimizeCenters:
                            S, boxes, mask = recenter(S, boxes, mask, ds)
                        if kk % 3 == 1 and thresh is not None:
                            S, activity, boxes, mask, L = mergeAll(S, activity, boxes, mask, L, ds)
                    else:
                        if kk % 20 == 0 and optimizeCenters:
                            S, boxes, mask = recenter(S, boxes, mask, ds)
                        if kk % 20 == 5 and thresh is not None:
                            S, activity, boxes, mask, L = mergeAll(S, activity, boxes, mask, L, ds)
                # S = HALS4shape(data0, S, activity)
            if it < len(iters0) - 1:
                mb = mbs[it + 1]
                data0 = data[:len(data) / mb * mb].reshape(-1, mb, prod(dims[1:])).mean(1)
                if D == 4:
                    data0 = data0.reshape(
                        len(data0), dims[1] / ds[0], ds[0], dims[2] / ds[1], ds[1], dims[3] / ds[2], ds[2])\
                        .mean(2).mean(3).mean(4)
                else:
                    data0 = data0.reshape(
                        len(data0), dims[1] / ds[0], ds[0], dims[2] / ds[1], ds[1]).mean(2).mean(3)
                data0.shape = (len(data0), -1)
                activity = ones((L + adaptBias, len(data0))) * activity.mean(1).reshape(-1, 1)
                activity = HALS4activity(data0, S, activity)

    ### Back to full data ##
        activity = ones((L + adaptBias, dims[0]),
                        dtype='float32') * activity.mean(1).reshape(-1, 1)
        if D == 4:
            S = repeat(repeat(repeat(S.reshape((-1,) + dims0[1:]),
                                     ds[0], 1), ds[1], 2), ds[2], 3).reshape(L + adaptBias, -1)
        else:
            S = repeat(repeat(S.reshape((-1,) + dims0[1:]),
                              ds[0], 1), ds[1], 2).reshape(L + adaptBias, -1)
        for ll in range(L):
            #     boxes[ll] = GetBox(centers[ll], R, dims[1:])
            boxes[ll] *= ds.reshape(-1, 1)
            temp = zeros(dims[1:])
            temp[map(lambda a: slice(*a), boxes[ll])] = 1
            mask[ll] = asarray(where(temp.ravel())[0])

        # from now on more iterations cause initial dot product in HALS is expensive for full data
        activity = HALS4activity(data, S, activity, 7)
        S = HALS4shape(data, S, activity, 7)


#### Main Loop ####
    skip = []
    for kk in range(iters):
        S, activity, skip = HALS(data, S, activity, skip, iters=10)  # , check_skip=kk)

        # Measure MSE
        if verbose:
            residual = data - activity.T.dot(S)
            MSE = dot(residual.ravel(), residual.ravel()) / data.size
            print('{0:1d}: MSE = {1:.5f}'.format(kk, MSE))
            if kk == (iters - 1):
                print('Maximum iteration limit reached')
            MSE_array.append(MSE)

    return asarray(MSE_array), S.reshape((-1,) + dims[1:]), activity, boxes


# example
# import numpy as np

# T = 50
# X = 200
# Y = 100
# data = np.random.randn(T, X, Y)
# centers = asarray([[40, 30]])
# data[:, 40, 30] += np.random.randn(T)
# sig = [3, 3]
# R = 3 * np.array(sig)
# dims = data.shape
# MSE_array, shapes, activity, boxes = LocalNMF(
#     data, centers, sig, NonNegative=True, verbose=True)
