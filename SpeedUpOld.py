from numpy import min, max, array, percentile, outer, zeros, dot, reshape, r_, ix_, arange, exp, nan_to_num
from time import time
import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt
from BlockGroupLasso import gaussian_group_lasso, GetCenters, GetROI, GetActivity


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


def LocalNMF(data, centers, sig, NonNegative=True,
             tol=1e-4, iters=10, verbose=False, adaptBias=True, iters0=0):
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
    R = 3 * array(sig)  # size of bounding box is 4 times size of neuron
    L = len(centers)
    shapes = []
    boxes = zeros((L, D - 1, 2), dtype=int)
    MSE_array = []
    activity = np.zeros((L, dims[0]))


# Initialize shapes, activity, and residual
    for ll in range(L):
        boxes[ll] = GetBox(centers[ll], R, dims[1:])
        temp = [(arange(dims[i + 1]) - centers[ll][i]) ** 2 / (2 * sig[i] ** 2)
                for i in range(D - 1)]
        temp = exp(-sum(ix_(*temp)))
        temp.shape = (1,) + dims[1:]
        temp = RegionCut(temp, boxes[ll])
        shapes.append(temp[0])
    residual = data.astype('float')
    if adaptBias:
         # Initialize background as 30% percentile
        # b_t = ones(len(residual))
        b_s = percentile(residual, 30, 0)  # .ravel()
        residual -= b_s
    # Initialize activity from strongest to weakest
    # based on data-background-stronger neurons and Gaussian shapes
    for ll in np.argsort([residual[:, c[0], c[1]].max() for c in centers])[::-1]:
        X = RegionCut(residual, boxes[ll])
        activity[ll] = dot(X, shapes[ll]) / dot(shapes[ll], shapes[ll])
        if NonNegative:
            activity[ll][activity[ll] < 0] = 0
        residual = RegionAdd(residual, -outer(activity[ll], shapes[ll]), boxes[ll])

    # (Re)calculate background based on data-neurons using nonnegative greedy PCA
    if adaptBias:
        residual += b_s
        residual.shape = (dims[0], -1)
        b_s = b_s.ravel()
        b_t = dot(residual, b_s) / dot(b_s, b_s)
        b_t[b_t < 0] = 0
        b_s = dot(residual.T, b_t) / dot(b_t, b_t)
        b_s[b_s < 0] = 0
        # res0 = residual.copy()
        residual -= outer(b_t, b_s)
        residual.shape = dims
        zz = b_t.mean()
        b_s *= zz
        b_t /= zz

    tls = [[time() - t, dot(residual.ravel(), residual.ravel())]]

#### Get shape estimates on subset of data ####
    if iters0 > 0:
        shapes0 = np.copy(shapes)
        T0 = np.arange(1000)  # timeindices for subsampling
        # T0 = np.arange(0,3000,3)
        # T0 = reduce(np.union1d, [np.argsort(a)[-20:] for a in activity])  # high
        # activity timepoints
        if adaptBias:
            b_t0 = b_t[T0].copy()
        # shapes0 = np.copy(shapes)
        activ = activity[:, T0]
        res = residual[T0].copy()
        for kk in range(iters0):
            # print 'subset', time() - t, kk
            for ll in range(L):
                # cut region and add neuron
                as0 = outer(activ[ll], shapes[ll])
                X = RegionCut(res, boxes[ll]) + as0

        # NonNegative greedy PCA
                for _ in range(2):
                    activ[ll] = nan_to_num(dot(X, shapes[ll]) / dot(shapes[ll], shapes[ll]))
                    if NonNegative:
                        activ[ll][activ[ll] < 0] = 0
                    shapes[ll] = nan_to_num(dot(X.T, activ[ll]) / dot(activ[ll], activ[ll]))
                    if NonNegative:
                        shapes[ll][shapes[ll] < 0] = 0

        # Update region
                res = RegionAdd(res, as0 - outer(activ[ll], shapes[ll]), boxes[ll])

        # Recalculate background
            if adaptBias:
                res.shape = (len(T0), -1)
                res += outer(b_t0, b_s)
                for _ in range(1):
                    b_s = dot(res.T, b_t0) / dot(b_t0, b_t0)
                    b_s[b_s < 0] = 0
                    b_t0 = dot(res, b_s) / dot(b_s, b_s)
                    b_t0[b_t0 < 0] = 0
                res -= outer(b_t0, b_s)
                res.shape = [len(T0)] + list(dims[1:])

    ### Back to full data ##
    # Update activities
    #     for ll in range(L):
    # cut region and add neuron
    #         X = RegionCut(residual, boxes[ll]) + outer(activity[ll], shapes0[ll])
    #         activity[ll] = nan_to_num(dot(X, shapes[ll]) / dot(shapes[ll], shapes[ll]))
    #     if NonNegative:
    #         activity[activity < 0] = 0
    # (Re)calculate background based on data-neurons using shape from subset
    #     residual = data.astype('float')
    #     for ll in range(L):
    #         residual = RegionAdd(residual, -outer(activity[ll], shapes[ll]), boxes[ll])
    #     if adaptBias:
    #         residual.shape = (dims[0], -1)
    #         b_t = dot(residual, b_s) / dot(b_s, b_s)
    #         b_t[b_t < 0] = 0
    #         residual -= outer(b_t, b_s)
    #         residual.shape = dims

        residual = data.astype('float')
        if adaptBias:
            b_s *= b_t0.mean()
            residual -= b_s.reshape(dims[1:])
        # Initialize activity from strongest to weakest
        # based on data-background-stronger neurons and Gaussian shapes
        for ll in np.argsort([residual[:, c[0], c[1]].max() for c in centers])[::-1]:
            X = RegionCut(residual, boxes[ll])
            activity[ll] = dot(X, shapes[ll]) / dot(shapes[ll], shapes[ll])
            if NonNegative:
                activity[ll][activity[ll] < 0] = 0
            residual = RegionAdd(residual, -outer(activity[ll], shapes[ll]), boxes[ll])

        # (Re)calculate background based on data-neurons using nonnegative greedy PCA
        if adaptBias:
            residual.shape = (dims[0], -1)
            residual += b_s
            b_s = b_s.ravel()
            b_t = dot(residual, b_s) / dot(b_s, b_s)
            b_t[b_t < 0] = 0
            b_s = dot(residual.T, b_t) / dot(b_t, b_t)
            b_s[b_s < 0] = 0
            residual -= outer(b_t, b_s)
            residual.shape = dims
            zz = b_t.mean()
            b_s *= zz
            b_t /= zz


# Update background first
#     residual = data.astype('float')
#     if adaptBias:
# b_s = dot(residual.T, b_t) / dot(b_t, b_t)
# b_s[b_s < 0] = 0
#         b_t = dot(resWB, b_s) / dot(b_s, b_s)
#         b_t[b_t < 0] = 0
# residual.shape = (dims[0], -1)
#     residual -= outer(b_t, b_s).reshape(dims)
# for ll in range(L):
# X = RegionCut(residual, boxes[ll])
# activity[ll] = dot(X, shapes[ll]) / dot(shapes[ll], shapes[ll])
# if NonNegative:
# activity[ll][activity[ll] < 0] = 0
#     for ll in np.argsort([residual[:, c[0], c[1]].max() for c in centers])[::-1]:
#         X = RegionCut(residual, boxes[ll])
#         activity[ll] = dot(X, shapes[ll]) / dot(shapes[ll], shapes[ll])
#         if NonNegative:
#             activity[ll][activity[ll] < 0] = 0
#         residual = RegionAdd(residual, -outer(activity[ll], shapes[ll]), boxes[ll])
# for ll in range(L):
# residual = RegionAdd(residual, -outer(activity[ll], shapes[ll]), boxes[ll])

# Estimate noise level
    def GetSnPSD(Y):
        L = len(Y)
        ff, psd_Y = welch(Y, nperseg=round(L / 8))
        sn = np.sqrt(np.mean(psd_Y[ff > .3] / 2))
        return sn
    noise = np.zeros(L)

#### Main Loop ####
    skip = []
    for kk in range(iters):
        # print 'main', time() - t, kk
        for ll in range(L):
            # if ll in skip:
            #     continue

            # cut region and add neuron
            as0 = outer(activity[ll], shapes[ll])
            X = RegionCut(residual, boxes[ll]) + as0
            # NonNegative greedy PCA
            for ii in range(3):
                activity[ll] = nan_to_num(dot(X, shapes[ll]) / dot(shapes[ll], shapes[ll]))
                if NonNegative:
                    activity[ll][activity[ll] < 0] = 0
                shapes[ll] = nan_to_num(dot(X.T, activity[ll]) / dot(activity[ll], activity[ll]))
                if NonNegative:
                    shapes[ll][shapes[ll] < 0] = 0
            as0 -= outer(activity[ll], shapes[ll])
            # if kk == 0:
            #     noise[ll] = GetSnPSD(activity[ll])
            # elif np.allclose(0, as0.mean(1), 1e-6, .05 * noise[ll]):
            #     skip += [ll]
            #     print 'skip', ll
            # Update region
            residual = RegionAdd(residual, as0, boxes[ll])
            # if ll==0: print '  RegionAdd', time() - t

        # Recalculate background
        if adaptBias:
            residual.shape = (dims[0], -1)
            residual += outer(b_t, b_s)
            for _ in range(1):
                b_s = dot(residual.T, b_t) / dot(b_t, b_t)
                b_s[b_s < 0] = 0
                b_t = dot(residual, b_s) / dot(b_s, b_s)
                b_t[b_t < 0] = 0
            residual -= outer(b_t, b_s)
            residual.shape = dims
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
        #                      lower[1] - newbox[1, 0]:upper[1] - newbox[1, 0]] = \
        #                 shp[lower[0] - boxes[ll][0, 0]:upper[0] - boxes[ll][0, 0],
        #                     lower[1] - boxes[ll][1, 0]:upper[1] - boxes[ll][1, 0]]
        #             shapes[ll] = newshape.reshape(-1)
        #             boxes[ll] = newbox

        # Measure MSE
        MSE = dot(residual.ravel(), residual.ravel())
        tls += [[time() - t, MSE]]
        if verbose:
            print('{0:1d}: MSE = {1:.3f}'.format(kk, MSE))
        if kk > 0 and abs(1 - MSE / MSE_array[-1]) < tol:
            break
        if kk == (iters - 1):
            print('Maximum iteration limit reached')
        MSE_array.append(MSE)
    if adaptBias:
        return tls, shapes, activity, boxes, outer(b_t, b_s).reshape(dims)
    else:
        return MSE_array, shapes, activity, boxes

########################################################

if __name__ == "__main__":
    # Fetch Data, take only 100x100 patch to not have to wait minutes
    sig = (4, 4)
    lam = 40
    data = np.asarray([np.load('../zebrafish/ROI_zebrafish/data/1/nparrays/TM0%04d_200-400_350-550_15.npy' % t)
                       for t in range(3000)])[:, :100, :100]
    x = np.load('x.npy')[:, :100, :100]  # x is stored result from grouplasso
    pic_x = np.percentile(x, 95, 0)
    cent = GetCenters(pic_x)

    MSE_array = [LocalNMF(data, (array(cent)[:-1]).T, sig, verbose=True, iters=5, iters0=i)[0]
                 for i in range(4)]
    plt.figure()
    for i, m in enumerate(MSE_array):
        plt.plot(*np.array(m).T, label=i)
    plt.legend(title='subset iterations')
    plt.xlabel('Walltime')
    plt.ylabel('MSE')
    plt.show()
