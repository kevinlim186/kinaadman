import numpy as np
import numba

@numba.njit()
def calculate_basic_features(X:np.ndarray, y:np.ndarray)->tuple:
    '''
    Calculate basic features of the function

    X: numpy array  containing the feature set
    y: numpy array  containing function objectives 

    returns list of numbers [observations, lower_min, lower_max, upper_min, upper_max, objective_min, objective_max]
    '''

    observations = len(X)
    
    maxes = []
    mins = []
    for x in X:
        max_=-1e-99
        min_=1e99
        for xs in x:
            if not max_ or xs>max_:
                max_=xs
            if not min_ or xs<min_:
                min_=xs
        maxes.append(max_)
        mins.append(min_)

    lower_min = min(mins)
    lower_max = max(mins)
    upper_min = min(maxes)
    upper_max = max(maxes)


    objective_min = np.min(y)
    objective_max = np.max(y)

    return [observations, lower_min, lower_max, upper_min, upper_max, objective_min, objective_max]
