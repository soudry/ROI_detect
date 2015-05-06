from os import getcwd
from numpy import zeros, reshape, r_, ix_


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
        box[dd, 0] = max(centers[dd] - R[dd], 0)
        box[dd, 1] = min(centers[dd] + R[dd] + 1, dims[dd])
    return box


def RegionAdd(Z, X, box):
    # Parameters:
    #  Z -  dataset [XxYxZ...]xT array
    #  box - Dx2 array defining spatial box to put X in
    #  X -  Input array (prod(diff(box,1))xT)

    # return:
    #  Z=Z+X on box region - a [XxYxZ...]xT array
    Z[ix_(*map(lambda a: range(*a), box))] += reshape(X,
                                                      (r_[box[:, 1] - box[:, 0], -1]))
    return Z


def RegionCut(X, box, *args):
    # CUTREGION Summary of this function goes here
    # Parameters:
    #  X -  an [XxYxZ...]xT array
    #  box - region to cut
    #  args - specificy dimensions of whole picture (optional)

    # return:
    #  res - Matrix of size prod(R)xT
    dims = X.shape
    if len(args) > 0:
        dims = args[0]
    if len(dims) - 1 != len(box):
        raise Exception('box has the wrong number of dimensions')
    return X[ix_(*map(lambda a: range(*a), box))].reshape((-1, dims[-1]))


# test python 3d:
# Z = np.ones((3, 4, 5, 5))
# X = np.arange(40).reshape((5, 8)).T
# box = np.array([[0, 2], [1, 3], [2, 4]])
# RegionAdd(Z, X, box)
# print Z[0, 1, 2, 3]
# print RegionCut(Z,box)
# test python 2d:
# Z = np.ones((6, 4, 5))
# X = np.arange(45).reshape((5, 9)).T
# box = np.array([[0, 3], [1, 4]])
# RegionAdd(Z, X, box)
# print Z[0, 1, 2]
# print RegionCut(Z,box)
# test matlab 3d:
# Z = ones(3, 4, 5, 5);
# X = reshape(0: 39, [8, 5]);
# box = [1, 2; 2, 3; 3, 4];
# Z = RegionAdd(Z, X, box);
# display(Z(1, 2, 3, 4));
# display(RegionCut(Z,box));
# test matlab 2d:
# Z = ones(6, 4, 5);
# X = reshape(0: 44, [9, 5]);
# box = [1, 3; 2, 4];
# Z = RegionAdd(Z, X, box);
# display(tmp(1, 2, 3));
# display(RegionCut(Z,box));
