from numpy import min, max, array, asarray, ravel, percentile, outer, zeros, ones, dot, \
    reshape, r_, ix_, arange, exp, nan_to_num, argsort, prod, mean, sqrt
from time import time
import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt
# from BlockGroupLasso import gaussian_group_lasso, GetCenters, GetROI, GetActivity
# from scipy.optimize import nnls # cvxopt is faster!
from cvxopt import matrix
from cvxopt import solvers
# import mosek
solvers.options['show_progress'] = False  # True
# solvers.options['MOSEK'] = {mosek.iparam.log: 0}


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
    R = 4 * array(sig)  # size of bounding box is 4 times size of neuron
    L = len(centers)
    shapes = []
    mask = []
    boxes = zeros((L, D - 1, 2), dtype=int)
    MSE_array = []
    mb = mbs[0] if iters0[0] > 0 else 1
    activity = zeros((L, dims[0] / mb))
    tls = []
    tsub = 0

### Initialize shapes, activity, and residual ###
    for ll in range(L):
        boxes[ll] = GetBox(centers[ll], R, dims[1:])
        temp = zeros(dims[1:])
        temp[map(lambda a: slice(*a), boxes[ll])]=1
        mask += np.where(temp.ravel())
        temp = [(arange(dims[i + 1]) - centers[ll][i]) ** 2 / (2 * sig[i] ** 2)
                for i in range(D - 1)]
        temp = exp(-sum(ix_(*temp)))
        temp.shape = (1,) + dims[1:]
        temp = RegionCut(temp, boxes[ll])
        shapes.append(temp[0])
    print 'init', time() - t
    # faster by factor 2 compared to keeping int16
    data = data.astype('float').reshape(dims[0], -1)
    residual = data.reshape((-1, mb) + dims[1:]).mean(1)
    dims0 = residual.shape
    data0 = residual.copy().reshape(len(residual), -1)
    if adaptBias:
         # Initialize background as 30% percentile
        b_s = percentile(residual, 30, 0)  # .ravel()
        residual -= b_s
    # Initialize activity from strongest to weakest
    # based on data-background-stronger neurons and Gaussian shapes
    for ll in argsort([residual[:, c[0], c[1]].max() for c in centers])[::-1]:
        X = RegionCut(residual, boxes[ll])
        activity[ll] = dot(X, shapes[ll]) / dot(shapes[ll], shapes[ll])
        if NonNegative:
            activity[ll][activity[ll] < 0] = 0
    # for ll in range(L):
        residual = RegionAdd(residual, -outer(activity[ll], shapes[ll]), boxes[ll])

    # (Re)calculate background based on data-neurons using nonnegative greedy PCA
    if adaptBias:
        residual += b_s
        residual.shape = (dims0[0], -1)
        b_s = b_s.ravel()
        b_t = dot(residual, b_s) / dot(b_s, b_s)
        b_t[b_t < 0] = 0
        b_s = dot(residual.T, b_t) / dot(b_t, b_t)
        b_s[b_s < 0] = 0
      # only if HALS using residual
        # residual -= outer(b_t, b_s)
        # residual.shape = dims0
        # zz = b_t.mean()
        # b_s *= zz
        # b_t /= zz
    S = zeros((L + adaptBias, prod(dims[1:])))
    for ll in range(L):
        S[ll] = RegionAdd(
            zeros((1,) + dims[1:]), shapes[ll].reshape(1, -1), boxes[ll]).ravel()
    if adaptBias:
        S[-1] = b_s
        activity = np.r_[activity, b_t.reshape(1, -1)]

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

  # HALS using residual
  #   def HALS(residual, shapes, activity, b_s, b_t, skip=[], check_skip=0):
  #       dims0 = residual.shape
  #       for ll in argsort(activity.max(1)):
  # for ll in range(L):
  #           if ll in skip:
  #               continue
  # cut region and add neuron
  #           if check_skip:
  #               a0 = activity[ll] / activity[ll].mean()
  #           as0 = outer(activity[ll], shapes[ll])
  #           X = RegionCut(residual, boxes[ll])
  # NonNegative greedy PCA
  #           shapes[ll] += nan_to_num(dot(X.T, activity[ll]) / dot(activity[ll], activity[ll]))
  #           if NonNegative:
  #               shapes[ll][shapes[ll] < 0] = 0
  #           activity[ll] += nan_to_num(dot(X, shapes[ll]) / dot(shapes[ll], shapes[ll]))
  #           if NonNegative:
  #               activity[ll][activity[ll] < 0] = 0
  # skip neurons whose shapes already converged
  #           if check_skip:
  # if check_skip == 1:  # compute noise level only once
  #                   noise[ll] = GetSnPSD(a0)
  #               if np.allclose(a0, activity[ll] / activity[ll].mean(), 1e-4, .01 * noise[ll]):
  #                   skip += [ll]
  # Update region
  #           residual = RegionAdd(residual, as0 - outer(activity[ll], shapes[ll]), boxes[ll])
  # Recalculate background
  #       if adaptBias:
  #           residual.shape = (dims0[0], -1)
  #           b0 = outer(b_t, b_s)
  #           b_s += dot(residual.T, b_t) / dot(b_t, b_t)
  #           b_s[b_s < 0] = 0
  #           b_t += dot(residual, b_s) / dot(b_s, b_s)
  #           b_t[b_t < 0] = 0
  #           residual += (b0-outer(b_t, b_s))
  #           residual.shape = dims0
  #       return residual, shapes, activity, b_s, b_t, skip

    def HALS4activity(data, S, activity):
        A = data.dot(S.T)
        B = S.dot(S.T)
        for ll in range(L + adaptBias):
            activity[ll] += nan_to_num((A[:, ll] - np.dot(activity.T, B[:, ll])) / B[ll, ll])
            if NonNegative:
                activity[ll][activity[ll] < 0] = 0
        tsub = time()
        residual = data - activity.T.dot(S)
        tsub -= time()
        return residual, activity, tsub

    # def HALS4activity(residual, shapes, activity, b_s, b_t):  # pass true residual
    #     dims0 = residual.shape
    # for ll in argsort(activity.max(1))[::-1]:
    #     for ll in range(L):
    # cut region and add neuron
    #         a0 = activity[ll].copy()
    #         X = RegionCut(residual, boxes[ll])
    # NonNegative greedy PCA
    #         activity[ll] += nan_to_num(dot(X, shapes[ll]) / dot(shapes[ll], shapes[ll]))
    #         if NonNegative:
    #             activity[ll][activity[ll] < 0] = 0
    # Update region
    #         residual = RegionAdd(residual, outer(a0 - activity[ll], shapes[ll]), boxes[ll])
    # Recalculate background
    #     if adaptBias:
    #         residual.shape = (dims0[0], -1)
    #         b_t0 = b_t.copy()
    #         b_t += dot(residual, b_s) / dot(b_s, b_s)
    #         b_t[b_t < 0] = 0
    #         residual += outer(b_t0 - b_t, b_s)
    #         residual.shape = dims0
    #     return residual, activity, b_t

  # Solve nonnegative least squares problem
    def NNLS4activity(data, shapes, b_s):  # pass data
        t0 = time()
        S = zeros((L + 1, prod(dims[1:])))
        for ll in range(L):
            S[ll] = RegionAdd(
                zeros((1,) + dims[1:]), shapes[ll].reshape(1, -1), boxes[ll]).ravel()
        S[-1] = b_s
        # http://abel.ee.ucla.edu/cvxopt/userguide/coneprog.html#quadratic-programming
        P = matrix(S.dot(S.T))
        G = matrix(0.0, (L + 1, L + 1))
        G[::L + 2] = -1.0
        h = matrix(0.0, (L + 1, 1))

        def nnls(y):
            q = matrix(-S.dot(y))
            result = solvers.qp(P, q, G, h)  # , solver='mosek')
            return ravel(result['x'])

        activity = asarray([nnls(d.ravel()) for d in data]).T
        # def nnls(y, init): #initial conditions did't speed up
        #     q = matrix(-S.dot(y))
        # result = solvers.qp(P, q, G, h, initvals=init)  # , solver='mosek')
        #     return ravel(result['x'])
        # activity = asarray(
        #     [nnls(d.ravel(), activity[:, i / mb].tolist() + [b_t[i / mb]]) for i, d in enumerate(residual)]).T
        # Subtract background and neurons
        print 'Time for cvx: ', time() - t0
        tsub = time()
        residual = data - activity.T.dot(S).reshape(data.shape)
        tsub -= time()
        b_t = activity[-1]
        activity = activity[:-1]
        return residual, activity, b_t, tsub

  # Solve nonnegative least squares problem
    def NNLS4shape(data, activity, b_t):  # pass data
        # P = matrix(activity.dot(activity.T))
        act = np.r_[activity, b_t.reshape(1, -1)]
        P = matrix(act.dot(act.T))
        G = matrix(0.0, (L + 1, L + 1))
        G[::L + 2] = -1.0
        h = matrix(0.0, (L + 1, 1))

        def nnls(i, y):  # add contraints to boxes
            q = matrix(-act.dot(y))
            Als = []
            for ll in range(L):
                if not ((boxes[ll][0, 0] <= i % dims[-2] < boxes[ll][0, 1]) and
                        (boxes[ll][1, 0] <= i / dims[-2] < boxes[ll][1, 1])):
                    Als += [ll]
            A = matrix(0.0, (len(Als), L + 1))
            for i, ll in enumerate(Als):
                A[i, ll] = 1
            b = matrix(0.0, (len(Als), 1))
            result = solvers.qp(P, q, G, h, A, b)
            return ravel(result['x'])
        S = asarray([nnls(i, d.ravel())
                     for i, d in enumerate(data.reshape(len(data), -1).T)]).T
        residual = data - act.T.dot(S).reshape(data.shape)
        b_s = S[-1]
        S = S[:-1]
        for ll in range(L):
            shapes[ll] = RegionCut(S[ll].reshape((1,) + dims[1:]), boxes[ll])[0]
        return residual, shapes, b_s

#

### Get shape estimates on subset of data ###

    if iters0[0] > 0:
        skip = []
        for it in range(len(iters0)):
            for kk in range(iters0[it]):
                print 'subset', time() - t, kk, skip
                # replace data0 by residual for HALS using residual
                _, S, activity, skip, dt = HALS(data0, S, activity, skip)
                tsub += dt
            if it < len(iters0) - 1:
                mb = mbs[it + 1]
                data0 = data.reshape(-1, mb, prod(dims[1:])).mean(1)
                activity = ones((L + adaptBias, len(data0))) * activity.mean(1).reshape(-1, 1)
                residual, activity, dt = HALS4activity(data0, S, activity)
                tsub += dt

    ### Back to full data ##

    # NNLS
        # residual, activity, b_t, dt = NNLS4activity(data, shapes, b_s)
        # tsub += dt
    # HALS
        #     residual = data - b_s.reshape(dims[1:])
        activity = ones((L + adaptBias, dims[0])) * activity.mean(1).reshape(-1, 1)
        for _ in range(3):
            # replace data0 by residual for HALS using residual
            residual, activity, dt = HALS4activity(data, S, activity)
            tsub += dt

        MSE = dot(residual.ravel(), residual.ravel())
        tls += [[time() - t + tsub, MSE]]
        if verbose:
            print('{0:1d}: MSE = {1:.3f}'.format(-1, MSE))
        MSE_array.append(MSE)

#

#### Main Loop ####
    skip = []
    for kk in range(iters):
        print 'main', time() - t, kk, tsub
        # , kk + 1)  # replace data by residual for HALS using residual
        residual, S, activity, skip, dt = HALS(data, S, activity, skip)
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
        return tls, shapes, activity, boxes  # , outer(b_t, b_s).reshape(dims)
    else:
        return MSE_array, shapes, activity, boxes

########################################################

# Fetch Data, take only 100x100 patch to not have to wait minutes
sig = (4, 4)
lam = 40
data = np.asarray([np.load('../zebrafish/ROI_zebrafish/data/1/nparrays/TM0%04d_200-400_350-550_15.npy' % t)
                   for t in range(3000)])  # [:, : 100, : 100]
x = np.load('x.npy')  # [:, : 100, : 100]  # x is stored result from grouplasso
pic_x = np.percentile(x, 95, 0)
cent = GetCenters(pic_x)


# iterls = np.outer([10, 20, 40, 60, 80], np.ones(2, dtype=int)) / 2
# iterls = [0, 10, 20, 40, 60, 80]
iterls = [80]
MSE_array = [LocalNMF(data, (array(cent)[:-1]).T, sig, verbose=True, iters=20, iters0=[i], mbs=[30])[0]
             for i in iterls]
plt.figure()
for i, m in enumerate(MSE_array):
    plt.plot(*np.array(m).T, label=iterls[i])
plt.legend(title='subset iterations')
plt.xlabel('Walltime')
plt.ylabel('MSE')
# plt.xlim(0, 10)
# plt.ylim(2.5e9, 2.55e9)
plt.show()

# tls, shapes, activity, boxes, background = LocalNMF(
#     data, (array(cent)[:-1]).T, sig, verbose=True, iters=1, iters0=[60], mbs=[30])
# for ll in range(len(shapes)):
# figure()
# imshow(shapes[ll].reshape(np.diff(boxes[ll], 1).ravel()))
# figure()
# imshow(shapes[-1].reshape(np.diff(boxes[-1], 1).ravel()))
