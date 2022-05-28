import numpy as np
import numba
from kinaadman.functions.utils import cast_to_float

@numba.njit()
def calculate_meta_models(X:np.ndarray, y:np.ndarray)->tuple:
    '''
    Calculate meta model features from linear and quadratic models
    
    X: numpy array  containing the feature set
    y: numpy array  containing function objectives 
   
    returns linear_rsquared, linear_x_intercept, linear_coef_min, linear_coef_max, linear_max_by_min, linear_pair_rsquared,  quad_rsquared, quad_max_by_min, quad_pair_rsquared
    '''
    y = cast_to_float(y)

    n=len(y)
    k=len(X[0])
    
    X_squared = np.square(X)
    
    linear_pair = pairwise_interaction(X)
    p=len(linear_pair[0])

    A =  np.zeros((X.shape[0], X.shape[1]+1), dtype=np.float64)
    A[:,:X.shape[1]]=X
    A[:,X.shape[1]]=np.ones(len(X), dtype=np.float64)

    A_pair = np.zeros((linear_pair.shape[0], linear_pair.shape[1]+1), dtype=np.float64)
    A_pair[:,:linear_pair.shape[1]]=linear_pair
    A_pair[:,linear_pair.shape[1]]=np.ones(len(linear_pair), dtype=np.float64)

    A_squared =  np.zeros((X_squared.shape[0], (X_squared.shape[1]*2)+1), dtype=np.float64)
    A_squared_pair = np.zeros((X_squared.shape[0], (X_squared.shape[1]*2)), dtype=np.float64)
    A_squared[:,:X_squared.shape[1]]=X
    A_squared[:,X_squared.shape[1]:-1]=X_squared
    A_squared_pair[:,:X_squared.shape[1]]=X
    A_squared_pair[:,X_squared.shape[1]:]=X_squared
    A_squared[:,-1]=np.ones(len(X_squared), dtype=np.float64)
    
    pair = pairwise_interaction(A_squared_pair)
    A_squared_pair =  np.zeros((pair.shape[0], pair.shape[1]+1), dtype=np.float64)
    A_squared_pair[:, :-1]=pair
    A_squared_pair[:,pair.shape[1]]=np.ones(len(pair), dtype=np.float64)


    linear_model  = np.linalg.lstsq(A,y)
    linear_pair_model = np.linalg.lstsq(A_pair,y)
    quad_model = np.linalg.lstsq(A_squared,y)
    quad_pair_model = np.linalg.lstsq(A_squared_pair,y)



    linear_intercepts = linear_model[0]
    linear_residual = linear_model[1][0]
    linear_pair_residual = linear_pair_model[1][0]

    quad_intercepts = quad_model[0][k:]
    quad_residual = quad_model[1][0]

    if len(quad_pair_model[1])==0:
        quad_pair_residual=0
    else:
        quad_pair_residual = quad_pair_model[1][0]

    linear_x_intercept = linear_intercepts[-1]
    linear_coef_min =  min(np.absolute(linear_intercepts[:-1]))
    linear_coef_max =  max(np.absolute(linear_intercepts[:-1]))
    linear_max_by_min = linear_coef_max/linear_coef_min


    quad_x_intercept = quad_intercepts[-1]
    quad_coef_min =  min(np.absolute(quad_intercepts[:-1])) 
    quad_coef_max =  max(np.absolute(quad_intercepts[:-1]))
    quad_max_by_min = quad_coef_max/quad_coef_min

    # compute for adjusted R squared
    linear_residual = 1 - linear_residual / (n * y.var())
    linear_rsquared = 1 - (1 - linear_residual)*(n-1)/(n-k-1) 
    linear_pair_residual = 1 - linear_pair_residual / (n * y.var())
    linear_pair_rsquared = 1 - (1 - linear_pair_residual)*(n-1)/(n-p-1) 
    quad_residual = 1 - quad_residual / (n * y.var())
    quad_rsquared = 1 - (1 - quad_residual)*(n-1)/(n-k-1) 
    quad_pair_residual = 1-quad_pair_residual/ (n * y.var())
    quad_pair_rsquared = 1 - (1 - quad_pair_residual)*(n-1)/(n-k-1) 

    return linear_rsquared, linear_x_intercept, linear_coef_min, linear_coef_max, linear_max_by_min, linear_pair_rsquared,  quad_rsquared, quad_max_by_min, quad_pair_rsquared
    
@numba.njit()
def pairwise_interaction(X:np.ndarray)->np.ndarray:
    row = len(X)
    col = len(X[0])
    new_cols = binomial(col,2)
    res = np.zeros((row,col+new_cols), dtype=np.float64)
    res[:,:col] = X
    
    n=col
    for i in range(col):
        for j in range(col):
            if j>i:
                res[:,n]=X[:,i]*X[:,j]
                n+=1

    return res       

@numba.njit()
def binomial(n, r):
    ''' Binomial coefficient, nCr, aka the "choose" function 
        n! / (r! * (n - r)!)
    '''
    p = 1    
    for i in range(1, min(r, n - r) + 1):
        p *= n
        p //= i
        n -= 1
    return p