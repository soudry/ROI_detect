from scipy.ndimage.measurements import label
from numpy import shape, ones, zeros, array, meshgrid, sqrt, sum, max, nanmean, nan


def GetROI(pic, cent):
    # find ROIs (regions of interest) for image 'pic' given centers -  by
    # choosing all the non-zero points nearest to each center
    BW_locations = pic > 0
    dims = shape(pic)
    components, _ = label(BW_locations, ones([3] * len(dims)))
    ROI = -ones(dims, dtype=int)
    mesh = meshgrid(indexing='ij', *map(range, dims))
    distances = array(
        [sqrt(sum((mesh[i] - c[i]) ** 2 for i in range(len(dims)))) for c in cent])
    min_dist_ind = distances.argmin(0)
    for ll in range(len(cent)):
        ind1 = components[tuple(cent[ll])]
        ind2 = min_dist_ind[tuple(cent[ll])]
        comp = (components == ind1)
        comp[min_dist_ind != ind2] = 0
        ROI[comp] = ll
    return ROI


def GetActivity(x, ROI):
    # Find activity from video x (size (XxYxZ...)xT ) given ROIs (regions of
    # intrest) by taking the spatial average in each region
    dims = shape(x)
    L = max(ROI) + 1
    activity = zeros((L, dims[-1]))
    ROI.shape = -1
    x = x.reshape(-1, dims[-1])
    x[ROI == -1] = nan
    for ll in range(L):
        activity[ll] = nanmean(x[ROI == ll], 0)
    return activity
