from functools import cache
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist,cdist
import numba
import warnings


@numba.njit(cache=True)
def calculate_dispersion(X:np.ndarray, y:np.ndarray, quantile:float=0.025, minimize:bool=True, central_tendency:str='mean', basis:str='diff', p:int=2)-> tuple[float, float, float, float]:
    '''
    Calculate dispersion

    X: numpy array  containing the feature set
    y: numpy array  containing function objectives 
    quantiles: float specifying the quantile of y values in consideration
    p: the p parameter of the minkowski distance. 1 is Manhattan distance 2 represent Euclidean distance.
    central_tendency: str choose between mean or mean 
    basis: str choose between diff or ratio
    '''

    assert central_tendency in ['mean', 'median'], "The basis selected must either be mean or mean ."
    assert basis in ['diff', 'ratio'], "The basis selected must either be mean or mean ."

    if len(X) >=5:

        if not minimize:
            y = y * -1

        q =np.quantile(y, quantile)
        condition = np.where(y<=q)

        xs = X[condition]
 
        if central_tendency=='mean':
            dist = pairwise_distance(xs,'mean',p)
            dist_full = pairwise_distance(X, 'mean',p)
        elif central_tendency=='median':
            dist = pairwise_distance(xs,'median',p)
            dist_full = pairwise_distance(X, 'median',p)


        if basis=='mean':
            if basis=='diff':
                return  dist - dist_full
            else:
                return  dist / dist_full

        else:
            if basis=='diff':
                return   dist - dist_full
            else:
                return  dist / dist_full

    else:
        return 0

    
@numba.njit(cache=True)
def pairwise_distance(X, central_tendency, p):
    m = X.shape[0]
    n = X.shape[1]

    D = np.empty((int(m * (m - 1) / 2), 1), dtype=np.float64)
    ind = 0

    for i in range(m):
        for j in range(i+1, m):
            d = 0.0

            for k in range(n):
                tmp = np.abs(X[i, k] - X[j, k])
                d += np.power(tmp, p)

            D[ind] = np.power(d, 1/p)
            ind += 1
    
    if central_tendency=='mean':
        return np.mean( D)
    elif central_tendency=='median':
        return np.median( D)