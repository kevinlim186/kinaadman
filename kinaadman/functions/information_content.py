from functools import cache
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist,cdist
import numba
import math


@numba.njit() 
def get_info_content(X:np.ndarray, y:np.ndarray):
    epsilon = [ np.power(10,i) for i in  np.linspace(-5, 15, num=1000)]
    X = np.unique(X, axis=1) #aggregate duplicates

    indices, dist = nearest_neighbor_path(X=X[:,:-1])
    
    res = compute_psi(indices=indices, x_distance=dist, y=y, eps=epsilon)
    


@numba.njit()
def nearest_neighbor_path(X:np.ndarray):
    #init values
    x = X[:,0]
    X = X[:,1:]
    
    indices =  np.empty(len(X[0])+1, dtype=np.int8)
    indices[0] = 0
    dist =  np.empty(len(X[0]))
    free_indices  = np.array([i for i in range(len(X[0]))], dtype=np.int8)
    

    for i in range(len(X[0])):
        min_ = get_nearest_neighbor(X=x, neighbors=X[:,free_indices])
        x= X[min_[0]]
        indices[i+1] = free_indices[min_[0]]
        dist[i] = min_[1]

        free_indices =  np.concatenate((free_indices[:min_[0]], free_indices[min_[0]+1:]))

    return indices, dist

@numba.njit()
def get_nearest_neighbor(X:np.ndarray, neighbors:np.ndarray):
    min_val=None
    min_index=0

    for i in range(len(neighbors[0])):
        distance =  np.sum(np.square(X-neighbors[:,i]))
        if min_val is None or min_val > distance:
            min_val=distance
            min_index=i
    
    return min_index, min_val

@numba.njit()
def compute_h(psi:np.ndarray):
    neg_neu = np.sum(np.where(psi[1:]==-1,0,True) & np.where(psi[:-1]==0,0,True))
    neg_pos = np.sum(np.where(psi[1:]==-1,0,True) & np.where(psi[:-1]==0,0,True))
    neu_neg = np.sum(np.where(psi[1:]==-1,0,True) & np.where(psi[:-1]==0,0,True))
    
    neu_pos = np.sum(np.where(psi[1:]==-1,0,True) & np.where(psi[:-1]==0,0,True))
    pos_neg = np.sum(np.where(psi[1:]==-1,0,True) & np.where(psi[:-1]==0,0,True))
    pos_neu = np.sum(np.where(psi[1:]==-1,0,True) & np.where(psi[:-1]==0,0,True))

    neg_neu = 0 if neg_neu==0 else neg_neu * math.log(neg_neu, 6)
    neg_pos = 0 if neg_pos==0 else neg_pos * math.log(neg_pos, 6)
    neu_neg = 0 if neu_neg==0 else neu_neg * math.log(neu_neg, 6)
    neu_pos = 0 if neu_pos==0 else neu_pos * math.log(neu_pos, 6)
    pos_neg = 0 if pos_neg==0 else pos_neg * math.log(pos_neg, 6)
    pos_neu = 0 if pos_neu==0 else pos_neu * math.log(pos_neu, 6)

    return neg_neu, neg_pos, neu_neg, neu_pos, pos_neg, pos_neu

@numba.njit()
def compute_m(psi:np.ndarray):
    n = len(psi)
    psi = psi[np.where(psi!=0)]
    psi = np.concatenate([np.array([False]), np.diff(psi)!=0])
    return len(psi) / (n - 1)

@numba.njit()
def compute_psi(indices:np.ndarray, x_distance:np.ndarray, y:np.ndarray, eps:np.ndarray):
    ratio = np.diff(y[indices])/x_distance
    res = np.array([])
    for e in eps:
        res = np.concatenate(res ,[0 if abs(x) > e else math.copysign(1, x) for x in ratio ])
     
    return res
