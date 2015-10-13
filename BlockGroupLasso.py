from __future__ import division

from numpy import zeros, ones, maximum, std, sum, nan_to_num, nanmean, nan,\
    mean, max, sqrt, percentile, dot, outer, asarray, meshgrid, argsort, zeros_like
from numpy.linalg import norm
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.measurements import label

""" See example script at the bottom.
Some comments:
The main function is
x=gaussian_group_lasso(data,...)
in which we operate on the data array and extract x,
which has the same dimensions like data, but is non-zero only
where neurons are detected. We assume that any "background" has been
subtracted already (e.g., rank-1 non negative matrix factorization components).

Later, by doing
y=std(x,0)
peaks=detect_peaks(y)
we output the neuronal centers from x.
This last bit is somewhat heuristically done using a regional max operation.
This can tend to over-split neurons. This seems better than using
connected components, which over-unify neurons (since these neural centers are
given as initialization to the non-negative matrix factorization script,
we don't want to be too conservative)

There are two main parameters:

1. sig - estimate radius of the neurons (e.g., 5 pixels)
2. lam - regularization parameter which controls the spatial sparsity.
Starting from some initial guess, lam can be automatically adjusted
by setting TargetRange to some range of expected spatial sparsity
(e.g., [0.05, 0.1] ). When I used this on a large dataset (e.g. whole brain)
I segement the dataset to patches and, adjust lam on a about 4
"representative" patches to get a single value which
gives reasonable performance in all patches. I then used this value of lam
on the whole brain (just set TargetRange=[] to avoind lam being adjusted).

One last comment: the algorithm works less well close to the edges
of the image ("close"~sig). I usually just throw away anything detected
near the edges and used overlapping patchs to compnesate.

"""


def gaussian_group_lasso(data, sig, lam=0.5, tol=1e-2, iters=100, NonNegative=True, TargetAreaRatio=[], verbose=False, adaptBias=False):
    """ Solve gaussian group lasso problem min_x 1/2*||Ax-data||_F^2 + lam*Omega(x)
        where Ax is convolution of x with a Gaussian filter A,
        and Omega is the group l1/l2 norm promoting spatial sparsity

        Input:
            data - Tx(XxYx...Z) array of the data
            sig - size of the gaussian kernel in different spatial directions
          Optional:
            lam  - regularization parameter (initial estimate)
            tol  - tolerance for stopping FISTA
            iters - maximum number of iterations
            NonNegative -  if true, neurons should be considered as non-negative
            TargetAreaRatio - if non-empty lamda is tuned so that the non-zero
            area fraction (sparsisty) of xk is between TargetAreaRatio[0]
            and TargetAreaRatio[1]
            verbose - print progress if true
            adaptBias - subtract rank 1 estimate of bias
        Output:
            xk - the final value of the iterative minimization
    """

    def A(data, do_transpose=False):
        if type(do_transpose) is bool:
            # Conveniently, the transpose of a gaussian filter matrix is a
            # gaussian filter matrix
            return gaussian_filter(data, (0,) + sig)
        elif type(do_transpose) is list:
            return gaussian_filter(data, tuple([sqrt(len(do_transpose)) * x for x in (0,) + sig]))
        else:
            raise NameError('do_transpose must be bool or list of bools')

    if NonNegative:
        # prox = lambda x, t: nan_to_num(maximum(1 - t / norm(maximum(x, 0),
        # ord=2, axis=0), 0) * maximum(x, 0))
        def prox(x, t):
            tmp = nan_to_num(  # faster and more memory efficent than numpy.linalg.norm
                maximum(1 - t / sqrt(sum((maximum(xx, 0) ** 2 for xx in x), axis=0)), 0))
            qq = zeros_like(x)
            for j, xx in enumerate(x):  # faster and more memory efficent than qq=tmp*maximum(x, 0)
                qq[j] = tmp * maximum(xx, 0)
            return qq
    else:
        # prox = lambda x, t: nan_to_num(maximum(1 - t / norm(x, ord=2, axis=0), 0) * x)
        def prox(x, t):
            tmp = nan_to_num(  # faster and more memory efficent than np.linalg.norm
                maximum(1 - t / sqrt(sum((xx ** 2 for xx in x), axis=0)), 0))
            qq = zeros_like(x)
            for j, xx in enumerate(x):  # faster and more memory efficent than qq=tmp*x
                qq[j] = tmp * xx
            return qq

    Omega = lambda x: sum(norm(x, ord=2, axis=0))
    # Lipshitz constant when Gaussian filter is normalized so it sums to 1
    L = 2

    if not TargetAreaRatio:
        return fista(data, prox, Omega, A, lam, L, tol=tol, NonNegative=NonNegative, iters=iters, verbose=verbose, adaptBias=adaptBias)
    else:  # Do exponential search to find lam
        lam_high = -1
        lam_low = -1
        rho = 10  # exponential search constant
        x = None

        while True:
            x = fista(data, prox, Omega, A, lam, L, x0=x, tol=tol,
                      NonNegative=NonNegative, iters=iters, verbose=verbose, adaptBias=adaptBias)
            y = mean(std(x, 0) > 0)
            print('Area Ratio = {0:.5f},lambda={1:.7f}'.format(y, lam))
            if y < TargetAreaRatio[0]:
                lam_high = lam
            elif y > TargetAreaRatio[1]:
                lam_low = lam
            else:
                return x
            if lam_high == -1:
                lam = lam * rho
            elif lam_low == -1:
                lam = lam / rho
            else:
                lam = (lam_high + lam_low) / 2


def fista(data, prox, Omega, A, lam, L, x0=None, tol=1e-8, iters=100, NonNegative=False, verbose=False, adaptBias=False):
    """ Fast Iterative Soft Threshold Algorithm for solving min_x 1/2*||Ax-data||_F^2 + lam*Omega(x)
        Input:
            data - matrix of the data B in the regularized optimization
            prox - proximal operator of the regularizer Omega
            Omega - regularizer
            A    - linear operator applied to x. The named argument 'do_transpose' determines
                   whether to apply the argument or its transpose. If a list of booleans is
                   given, the corresponding operators are applied in order (if possible)
            lam  - regularization parameter
            L    - Lipschitz constant. Should be the 2*(the largest eigenvalue of A^T*A).
          Optional:
            x0   - Initialization of x
            tol  - tolerance
            iters - maximum number of iterations
            NonNegative - NonNegative -  if true, neurons should be considered
            as non-negative
            verbose - print progress if true
            adaptBias - subtract rank 1 estimate of bias
        Output:
            xk - the final value of the iterative minimization
    """
    tk1 = 1
    sz = data.shape
    if x0 is None:
        from scipy.ndimage.filters import gaussian_filter, median_filter
        x0 = zeros(sz, dtype='float32')  # 419459.4
        # x0 = data * (data>percentile(data,99.9)) #417182.4
        # x0 = data * (data>percentile(data,99.5)) #418335.9
        # x0 = data * asarray([data[i] > percentile(data[i], 99.9) for i in range(len(data))])
        # x0 = asarray([median_filter(d,5) for d in data])
        # x0 *= asarray([x0i > percentile(x0i, 99) for x0i in x0]) #416122.1
        # x0 *= (x0 > percentile(x0, 99)) #416202.2
        # x0 = asarray([gaussian_filter(d,2) for d in data])
        # x0 *= (x0 > percentile(x0, 99)) #415899.2
    yk = x0
    xk = x0
    v = (2 / L * A(data.astype('float32'), do_transpose=True))
    del x0

    if adaptBias:
        b_t, b_s = greedyNNPCA(data, percentile(data, 30, 0).ravel().astype('float32'), 3)

    for kk in range(iters):
        xk1 = xk
        tk = tk1
        if adaptBias:
            r = A(yk, do_transpose=False)
            qk = - 2 / L * A(r + outer(b_t, b_s).reshape(sz), do_transpose=True) + v
            if kk % 5 == 4:
                b_t, b_s = greedyNNPCA(data - r, b_s, 3)
        else:
            qk = - 2 / L * A(yk, do_transpose=[False, True]) + v
        xk = prox(yk + qk, lam / L)
        tk1 = (1 + sqrt(1 + 4 * (tk ** 2))) / 2
        yk = (xk + (tk - 1) / tk1 * (xk - xk1))

        # Adaptive restart from Donoghue2012
        if dot(qk.ravel(), (xk - xk1).ravel()) > 0:
            tk1 = tk
            yk = xk

        norm_xk = norm(xk)
        if norm_xk == 0 or norm(xk - xk1) < tol * norm(xk1):
            return xk

        if verbose:
            resid = A(xk, do_transpose=False) - data + \
                (0 if not adaptBias else outer(b_t, b_s).reshape(sz))
            resid.shape = (sz[0], -1)
            loss = norm(resid, ord='fro') ** 2
            reg = Omega(xk)
            print('{0:1d}: Obj = {1:.1f}, Loss = {2:.1f}, Reg = {3:.1f}, Norm = {4:.1f}'.format(
                kk, 0.5 * loss + lam * reg, loss, reg, norm_xk))
    return xk


def greedyNNPCA(data, v_s, iterations):
    d = data.reshape(len(data), -1)
    v_s[v_s < 0] = 0
    for _ in range(iterations):
        v_t = dot(d, v_s) / sum(v_s ** 2)
        v_t[v_t < 0] = 0
        v_s = dot(d.T, v_t) / sum(v_t ** 2)
        v_s[v_s < 0] = 0
    return v_t, v_s


def GetROI(pic, cent):
    # find ROIs (regions of interest) for image 'pic' given centers -  by
    # choosing all the non-zero points nearest to each center
    dims = pic.shape
    components, _ = label(pic > 0, ones([3] * len(dims)))
    ROI = -ones(dims, dtype=int)
    mesh = meshgrid(indexing='ij', *map(range, dims))
    distances = asarray(
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
    # Find activity from video x (size Tx(XxYxZ...)) given ROIs (regions of
    # intrest) by taking the spatial average in each region
    dims = x.shape
    L = max(ROI) + 1
    activity = zeros((L, dims[0]))
    ROI.shape = -1
    x = x.reshape(dims[0], -1)
    x[:, ROI == -1] = nan
    for ll in range(L):
        activity[ll] = nanmean(x[:, ROI == ll], 1)
    return activity


def GetCenters(image):
    """ Take a image and detect the peaks using local maximum filter.
    input: a 2D image
    output: peaks list, in which
        peaks[0] - y coordinates
        peaks[1] - x coordinates
        peaks[2] - magnitude ("height") of peak
    """
    from skimage.feature import peak_local_max
    peaks = peak_local_max(image, min_distance=3, threshold_rel=.03).T
    magnitude = image[list(peaks)]
    indices = argsort(magnitude)[::-1]
    peaks = list(peaks[:, indices]) + [magnitude[indices]]
    return peaks
