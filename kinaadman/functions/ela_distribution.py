from curses.ascii import NL
import numpy as np
import numba
import math

@numba.njit()
def calculate_ela_distribution(y:np.ndarray)->tuple:
    '''
    Calculate distribution measures. 

    y: numpy array  containing function objectives 

    returns list of numbers [kur, skew, num_peakness]
    '''
    
    kur = kurtosis(y)
    skew = skewness(y)
    peaks = get_num_peaks(y)

    return kur, skew, peaks

@numba.njit()
def kurtosis(X:np.ndarray)->float:
    r=4 ## Based on MINITAB and BMDP or type 3 of e1071.kurtosis
    mean = np.mean(X)
    std = np.std(X)
    n=len(X)
    m = np.sum(np.power((X-mean),r)/n)
    g2 =  m/np.power(std,4)-3
    kurt = (g2+3)*np.power((1-1/n),2)-3

    return kurt

@numba.njit()
def skewness(X:np.ndarray)->float:
    r=3 ## Based on MINITAB and BMDP or type 3 of e1071.kurtosis
    mean = np.mean(X)
    std = np.std(X)
    n=len(X)
    m = np.sum(np.power((X-mean),r)/n)
    g1 = m/np.power(std,3)
    skewness = g1*np.power(((n-1)/n),3/2)

    return skewness
    
density = np.histogram(y, density=True)[1]
n = len(density)

@numba.jit(nopython=True)
def gaussian(x):
    '''
    From https://numba.pydata.org/numba-examples/examples/density_estimation/kernel/results.html 
    '''
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

@numba.jit(nopython=True, parallel=True)
def kde(samples:np.array)->tuple:
    '''
    From https://numba.pydata.org/numba-examples/examples/density_estimation/kernel/results.html 
    '''
    bandwidth = hsj(samples)
    eval_points = np.linspace(np.min(samples)-3*bandwidth, np.max(samples)+3*bandwidth, 512)
    result = np.zeros_like(eval_points)

    for i in numba.prange(len(eval_points)):
        eval_x = eval_points[i]
        for sample in samples:
            result[i] += gaussian((eval_x - sample) / bandwidth) / bandwidth
        result[i] /= len(samples)

    return result, eval_points

@numba.jit(nopython=True)
def get_num_peaks(X:np.array)->np.array:
    y, eval_points=kde(X)
    
    min_idx =  np.where((y[1:-1]<y[:-2])&(y[1:-1]<y[2:]))[0]
    threshold=0.01
    peaks=0

    v = np.zeros(len(min_idx)+2) 
    v[0] = 0
    v[-1] = len(y)-1
    v[1:len(min_idx)+1]= min_idx

    for i in range(len(v)-1):
        s_idx= int(v[i]+1)
        e_idx= int(v[i+1]+1)-1
        mean = np.mean(y[s_idx:e_idx+1])
        diff = eval_points[e_idx] - eval_points[s_idx] 
        modemass = mean * diff
        if modemass>=threshold:
            peaks+=1

    return peaks


@numba.jit(nopython=True)
def wmean(x, w):
    '''
    Weighted mean

    implementation of https://github.com/Neojume/pythonABC/blob/master/hselect.py
    '''
    return sum(x * w) / float(sum(w))

@numba.jit(nopython=True)
def wvar(x, w):
    '''
    Weighted variance

    implementation of https://github.com/Neojume/pythonABC/blob/master/hselect.py
    '''
    return sum(w * (x - wmean(x, w)) ** 2) / float(sum(w) - 1)



@numba.jit(nopython=True)
def normpdf(x:np.array, mean:float, sd:float):
    denom = (2*math.pi)**.5 * sd
    num = np.exp(-0.5 * ((x-mean)/(sd))**2)
    return num/denom

@numba.jit(nopython=True)
def sj(x, h):
    '''
    Equation 12 of Sheather and Jones [1]_
    References
    ----------
    .. [1] A reliable data-based bandwidth selection method for kernel
        density estimation. Simon J. Sheather and Michael C. Jones.
        Journal of the Royal Statistical Society, Series B. 1991

    implementation of https://github.com/Neojume/pythonABC/blob/master/hselect.py
    '''
    n = len(x)
    one = np.ones((1, n))

    lam = np.percentile(x, 75) - np.percentile(x, 25)
    a = 0.92 * lam * n ** (-1 / 7.0)
    b = 0.912 * lam * n ** (-1 / 9.0)

    W = tile(x)
    W = W - W.T

    W1 = phi6(W / b)
    tdb = np.dot(np.dot(one, W1), one.T)
    tdb = -tdb / (n * (n - 1) * b ** 7)

    W1 = phi4(W / a)
    sda = np.dot(np.dot(one, W1), one.T)
    sda = sda / (n * (n - 1) * a ** 5)

    alpha2 = 1.357 * (np.abs(sda / tdb)) ** (1 / 7.0) * h ** (5 / 7.0)

    W1 = phi4(W / alpha2)
    sdalpha2 = np.dot(np.dot(one, W1), one.T)
    sdalpha2 = sdalpha2 / (n * (n - 1) * alpha2 ** 5)

    return (normpdf(0, 0, np.sqrt(2)) /
            (n * np.abs(sdalpha2[0, 0]))) ** 0.2 - h

@numba.jit(nopython=True)
def phi6(x):
    return (x ** 6 - 15 * x ** 4 + 45 * x ** 2 - 15) * normpdf(x,0,1)


@numba.jit(nopython=True)
def phi4(x):
    return (x ** 4 - 6 * x ** 2 + 3) * normpdf(x,0,1)


@numba.jit(nopython=True)
def hnorm(x, weights=None):
    '''
    Bandwidth estimate assuming f is normal. See paragraph 2.4.2 of
    Bowman and Azzalini[1]_ for details.
    References
    ----------
    .. [1] Applied Smoothing Techniques for Data Analysis: the
        Kernel Approach with S-Plus Illustrations.
        Bowman, A.W. and Azzalini, A. (1997).
        Oxford University Press, Oxford

    implementation of https://github.com/Neojume/pythonABC/blob/master/hselect.py
    '''

    x = np.asarray(x)

    if weights is None:
        weights = np.ones(len(x))

    n = float(sum(weights))

    sd = np.sqrt(wvar(x, weights))
    return sd * (4 / (3 * n)) ** (1 / 5.0)


@numba.jit(nopython=True)
def hsj(x, weights=None):
    '''
    Sheather-Jones bandwidth estimator [1]_.
    References
    ----------
    .. [1] A reliable data-based bandwidth selection method for kernel
        density estimation. Simon J. Sheather and Michael C. Jones.
        Journal of the Royal Statistical Society, Series B. 1991

    implementation of https://github.com/Neojume/pythonABC/blob/master/hselect.py
    '''

    h0 = hnorm(x)
    v0 = sj(x, h0)

    if v0 > 0:
        hstep = 1.1
    else:
        hstep = 0.9

    h1 = h0 * hstep
    v1 = sj(x, h1)

    while v1 * v0 > 0:
        h0 = h1
        v0 = v1
        h1 = h0 * hstep
        v1 = sj(x, h1)

    return h0 + (h1 - h0) * abs(v0) / (abs(v0) + abs(v1))

@numba.jit(nopython=True)
def tile(x):
    n = len(x)
    res = np.zeros((n,n))

    for i in range(n):
        res[i] = x

    return res