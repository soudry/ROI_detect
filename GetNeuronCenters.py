from numpy import zeros, maximum, transpose, reshape, prod, std, sum, nan_to_num, mean,min,max
from numpy.linalg import norm
from scipy.ndimage.filters import gaussian_filter, maximum_filter
from scipy.ndimage.morphology import generate_binary_structure
from math import sqrt
import matplotlib.pyplot as plt


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

def gaussian_group_lasso(data, sig, lam=0.5, tol=1e-7, iters=100,NonNegative=False,TargetAreaRatio=[], verbose=False):
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
        Output:
            xk - the final value of the iterative minimization """    
    
    def A(data,do_transpose=False):
        if type(do_transpose) is bool:
            # Conveniently, the transpose of a gaussian filter matrix is a gaussian filter matrix
            return gaussian_filter(data,(0,) + sig,mode='constant')
        elif type(do_transpose) is list:
            return gaussian_filter(data,tuple([sqrt(len(do_transpose))*x for x in (0,) + sig]),mode='wrap')
        else:
            raise NameError('do_transpose must be bool or list of bools')
            
    if NonNegative==True:
        prox = lambda x,t: nan_to_num(maximum(1-t/norm(maximum(x,0),ord=2,axis=0),0)*maximum(x,0))    
    else:                
        prox = lambda x,t: nan_to_num(maximum(1-t/norm(x,ord=2,axis=0),0)*x)
        
    Omega= lambda x:   sum(norm(x,ord=2,axis=0))
    L=2; #Lipshitz constant when Gaussian filter is normalized so it sums to 1
        
        
    if not TargetAreaRatio:
        return fista(data, prox, Omega,A, lam, L, tol=tol,NonNegative=NonNegative, iters=iters, verbose=verbose)
    else:
        lam_high=lam*1e2;
        lam_low=lam*1e-3;
        cond=True       
        x=None
        
        while cond:            
            x=fista(data, prox, Omega,A, lam, L,x0=x,tol=tol,NonNegative=NonNegative, iters=iters, verbose=verbose)
            y=mean(std(x,0)>0)
            print('Area Ratio = {0:.5f},lambda={1:.7f}'.format(y,lam))
            
            if y<TargetAreaRatio[0]:
                lam_high=lam
            elif  y>TargetAreaRatio[1]:
                lam_low=lam
            else:
                return x
                cond=False    
            lam=(lam_high+lam_low)/2
            

def fista(data, prox, Omega, A, lam, L, x0=None, tol=1e-6, iters=100,NonNegative=False, verbose=False):
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
        Output:
            xk - the final value of the iterative minimization """
    tk1 = 1
    if x0 is None:
        x0 = zeros(A(data,do_transpose=True).shape)
    sz = x0.shape
    yk = x0
    xk  = x0
    v = 2/L*A(data,do_transpose=True)
    for kk in range(iters):
        xk1 = xk
        tk  = tk1
        
        qk= yk - 2/L*A(yk,do_transpose=[False,True]) + v
        xk  = prox(qk, lam/L)
        
        if mean(xk-xk1)<tol:
            return xk
        
        tk1 = (1+sqrt(1+4*(tk**2)))/2
        yk = xk + (tk-1)/tk1*(xk-xk1)       
        
        do_restart=sum((qk-yk)*(xk-xk1))>0  # Adaptive restart from Donoghue2012 
        if do_restart:
            tk1=tk
            yk=xk
        
        if verbose:
            resid = A(xk,do_transpose=False)-data
            if len(sz) > 2:
                resid = reshape(resid,(sz[0],prod(sz[1:])))
            loss = norm(resid,ord='fro')**2
            reg  = Omega(xk)
            print('{0:1d}: Obj = {1:.1f}, Loss = {2:.1f}, Reg = {3:.1f}, Norm = {4:.1f}'.format(kk, 0.5*loss + lam*reg, loss, reg, norm(xk)))
    return xk


def detect_peaks(image):
    """
    Takes a image and detect the peaks using local maximum filter.
    input: a 2D image
    output: peaks list, in which 
        peaks[0] - y coordinates
        peaks[1] - x coordinates
        peaks[2] - magnitude ("height") of peak
    """

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2,2)

    #apply the local maximum filter; all pixel of maximal value 
    #in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood)==image
    #local_max is a mask that contains the peaks we are 
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.
    ind=image==0
    local_max[ind]=0
    
    width = len(local_max[0])
    peaks_x = []
    peaks_y= []
    magnitude=[]
    posn = 0
    for row in local_max:
        for col in row:
            if col ==1:
                y=posn // width -1
                x=posn % width -1
                peaks_y.append(y)
                peaks_x.append(x)
                magnitude.append(image[y,x])
            posn += 1
           
    indices=sorted(range(len(magnitude)), key=lambda k: magnitude[k])
    indices=indices[::-1]
    peaks_x=[peaks_x[ii] for ii in indices]
    peaks_y=[peaks_y[ii] for ii in indices]
    magnitude=[magnitude[ii] for ii in indices]
    peaks=[peaks_y, peaks_x,magnitude]
    
    return peaks
    
    #### Example Script

if __name__ == "__main__":
    from numpy.random import rand, randn, randint
    from numpy import eye
    import time
    import scipy.io
    import pylab

    data_source=2
    
    if data_source==1: # generate 2D model data        
        T = 100
        sz = (100,100)      
        sig = (5,5) #neurons size
        foo = 0.1*randn(*((T,) + sz))
        bar = zeros((T,) + sz)
        for i in range(20):
            ind = tuple([randint(x) for x in sz])
            for j in range(T):
                bar[(j,)+ind] = randn()
        data = foo + 10*gaussian_filter(bar,(0,)+sig)
    elif data_source==2:   # Use experimental 2D data  
        mat = scipy.io.loadmat('Datasets/data_exp2D')
        data=transpose(mat['data'],[2, 0, 1])
        sig = (5,5) #neurons size
        
    TargetRange=[0.03, 0.04]    
    lam=1;
#    TargetRange=[] 
    
    x=gaussian_group_lasso(data,sig,lam,NonNegative=True,TargetAreaRatio=TargetRange,verbose=True)
    y=std(x,0)
    peaks=detect_peaks(y)    
    z=std(data,0)


# Plot

    ax=plt.subplot2grid((1,2), (0,0), colspan=1)
    ax.imshow(z)
    plt.hold(True)
    ax.scatter(peaks[1],peaks[0],s=2*sig[1],marker='x',c='black')   
#    ax.scatter(peaks[1],peaks[0],s=2*sig[1],marker='o',c='white')   
    ax.set_title('Data with detected centers')
    ax2=plt.subplot2grid((1,2), (0,1), colspan=1)    
    ax2.imshow(y) 
    ax2.set_title('Inferred x')
    
    
    ## Video
    dt=1e-2;
    fig=plt.figure()
    ax=fig.add_subplot(111)
    mi=min(data)
    ma=max(data)
    for ii in range(data.shape[0]):        
        time.sleep(dt)        
        ax.set_title('Data with detected centers')
        ax.scatter(peaks[1],peaks[0],s=2*sig[1],marker='o',c='white') 
        plt.hold(True)
#        ax.scatter(peaks[1],peaks[0],s=3*sig[1],marker='x',c='black')           
#        plt.hold(True)
        ax.imshow(data[ii,:,:],vmin=mi, vmax=ma,aspect='auto')         
        pylab.draw()  
        plt.hold(False) 

