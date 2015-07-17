from numpy import sum, zeros, ones, array, reshape, r_, ix_, exp, arange,\
    dot, outer, nan_to_num, percentile
# from scipy.ndimage.measurements import center_of_mass


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

    temp = list(map(lambda a: range(*a), box))
    Z[ix_(*([range(len(Z))] + temp))] += reshape(X, (r_[-1, box[:, 1] - box[:, 0]]))
    return Z


def RegionCut(X, box, *args):
    # Parameters
    #  X : array, shape (T, X, Y[, Z])
    #  box : array, shape (D, 2), region to cut
    #  args : tuple, specificy dimensions of whole picture (optional)
    # Returns
    #  res : array, shape (T, prod(diff(box,1))),
    dims = X.shape
    if len(args) > 0:
        dims = args[0]
    if len(dims) - 1 != len(box):
        raise Exception('box has the wrong number of dimensions')
    return X[ix_(*([list(range(dims[0]))] + list(map(lambda a: range(*a), box))))].reshape((dims[0], -1))


def LocalNMF(data, centers, activity, sig, NonNegative=False,
             tol=1e-6, iters=100, verbose=False, adaptBias=False):
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
    # Initialize Parameters
    dims = data.shape
    D = len(dims)
    R = 4 * array(sig)  # size of bounding box is 4 times size of neuron
    L = len(centers)
    shapes = []
    boxes = zeros((L, D - 1, 2), dtype=int)
    MSE_array = []
    residual = data

# Initialize shapes, activity, and residual
    for ll in range(L):
        boxes[ll] = GetBox(centers[ll], R, dims[1:])
        temp = [(arange(dims[i + 1]) - centers[ll][i]) ** 2 / (2 * sig[i])
                for i in range(D - 1)]
        temp = exp(-sum(ix_(*temp)))
        temp.shape = (1,) + dims[1:]
        temp = RegionCut(temp, boxes[ll])
        shapes.append(temp[0])
    residual = data.copy()
# Initialize background as 30% percentile
    if adaptBias:
        b_t = ones(len(residual))
        b_s = percentile(residual, 30, 0).ravel()
        residual -= outer(b_t, b_s).reshape(dims)
# Initialize activity, iteratively remove background
    for _ in range(5):
        # (Re)calculate activity based on data-background and Gaussian shapes
        for ll in range(L):
            X = RegionCut(residual, boxes[ll])
            activity[ll] = dot(X, shapes[ll]) / dot(shapes[ll], shapes[ll])
            if NonNegative:
                activity[ll][activity[ll] < 0] = 0
    # (Re)calculate background based on data-neurons using nonnegative greedy PCA
        residual = data.copy()
        for ll in range(L):
            residual = RegionAdd(residual, -outer(activity[ll], shapes[ll]), boxes[ll])
        if not adaptBias:
            break
        residual.shape = (dims[0], -1)
        b_s = dot(residual.T, b_t) / dot(b_t, b_t)
        b_s[b_s < 0] = 0
        b_t = dot(residual, b_s) / dot(b_s, b_s)
        b_t[b_t < 0] = 0
        residual -= outer(b_t, b_s)
        residual.shape = dims

# Main Loop
    for kk in range(iters):
        for ll in range(L):
            # Add region
            residual = RegionAdd(
                residual, outer(activity[ll], shapes[ll]), boxes[ll])

            # Cut region
            X = RegionCut(residual, boxes[ll])

            # NonNegative greedy PCA
            for ii in range(5):
                activity[ll] = nan_to_num(dot(X, shapes[ll]) / dot(shapes[ll], shapes[ll]))
                if NonNegative:
                    activity[ll][activity[ll] < 0] = 0
                shapes[ll] = nan_to_num(dot(X.T, activity[ll]) / dot(activity[ll], activity[ll]))
                if NonNegative:
                    shapes[ll][shapes[ll] < 0] = 0

            # Subtract region
            residual = RegionAdd(residual, -outer(activity[ll], shapes[ll]), boxes[ll])

        # Recalculate background
        if adaptBias:
            residual = data.copy()
            for ll in range(L):
                residual = RegionAdd(residual, -outer(activity[ll], shapes[ll]), boxes[ll])
            residual.shape = (dims[0], -1)
            for _ in range(5):
                b_s = nan_to_num(dot(residual.T, b_t) / dot(b_t, b_t))
                b_s[b_s < 0] = 0
                b_t = nan_to_num(dot(residual, b_s) / dot(b_s, b_s))
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
        if verbose:
            print('{0:1d}: MSE = {1:.3f}'.format(kk, MSE))
        if kk > 0 and abs(1 - MSE / MSE_array[-1]) < tol:
            break
        if kk == (iters - 1):
            print('Maximum iteration limit reached')
        MSE_array.append(MSE)
    if adaptBias:
        return MSE_array, shapes, activity, boxes, outer(b_t, b_s).reshape(dims)
    else:
        return MSE_array, shapes, activity, boxes


# example
import numpy as np

T = 50
X = 200
Y = 100
data = np.random.randn(T, X, Y)
centers = [[40, 30]]
activity = [np.random.randn(T)]
sig = [3, 3]
R = 3 * np.array(sig)
dims = data.shape
MSE_array, shapes, activity, boxes = LocalNMF(
    data, centers, activity, sig, NonNegative=True)
