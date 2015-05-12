from os import getcwd
from numpy import sum, zeros, array, reshape, r_, ix_, exp, arange, dot, outer, prod


def GetHomeFolder():
    # Obtains the folder in which data is saved - change accordingly
    return getcwd()


def GetFileName(params):
    # generate file names for saving  (important parameters should appear in
    # name)
    return 'Saved_Results_sigma=' + str(params['sigma_vector'][0]) + '_lambda0=' + str(params['lambda'])


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
    Z[ix_(*([range(len(Z))] + map(lambda a: range(*a), box)))] += reshape(X,
                                                                          (r_[-1, box[:, 1] - box[:, 0]]))
    return Z


def RegionCut(X, box, *args):
    # CUTREGION Summary of this function goes here
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
    return X[ix_(*([range(dims[0])] + map(lambda a: range(*a), box)))].reshape((dims[0], -1))


def LocalNMF(data, centers, activity, sig, NonNegative=False, tol=1e-7, iters=100, verbose=False):
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
    R = 3 * array(sig)  # size of bounding box is 3 times size of neuron
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

        residual = RegionAdd(
            residual, -outer(activity[ll], shapes[ll]), boxes[ll])

# Main Loop
    for kk in range(iters):
        for ll in range(L):
            # add region
            residual = RegionAdd(
                residual, outer(activity[ll], shapes[ll]), boxes[ll])

            # cut region
            X = RegionCut(residual, boxes[ll])

            # NonNegative greedy PCA
            greedy_pca_iterations = 5
            for ii in range(greedy_pca_iterations):
                temp = dot(X, shapes[ll]) / sum(shapes[ll] ** 2)
                if NonNegative:
                    temp[temp < 0] = 0
                activity[ll] = temp

                temp = dot(X.T, activity[ll]) / sum(activity[ll] ** 2)
                if NonNegative:
                    temp[temp < 0] = 0
                shapes[ll] = temp

            # Subtract region
            residual = RegionAdd(
                residual, -outer(activity[ll], shapes[ll]), boxes[ll])

        # Measure MSE
        MSE = sum(residual ** 2) / prod(dims)
        if kk > 0 and abs(1 - MSE / MSE_array[-1]) < tol:
            break
        if kk == (iters - 1):
            print('Maximum iteration limit reached')
        MSE_array.append(MSE)
        if verbose:
            print('{0:1d}: MSE = {1:.3f}'.format(kk, MSE))

    return MSE_array, shapes, activity, boxes


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
