import numpy as np
import numba

@numba.njit()
def calculate_principal_component(X:np.ndarray, y:np.ndarray, threshold:float=0.9)->tuple:
    '''
    Calculate principal component of the x values

    X: numpy array  containing the feature set
    y: numpy array  containing function objectives 

    returns list of numbers [cov_x, cor_x, cov_init, cor_init, expl_var_PC1_cov_x, expl_var_PC1_cor_x, expl_var_PC1_cov_init, expl_var_PC1_cor_init]
    '''

    init =  np.zeros((X.shape[0], X.shape[1]+1))
    init[:,:X.shape[1]]=X
    init[:,X.shape[1]]=y

    X_meaned = X - np.mean(X )
    init_mean =  init - np.mean(init)
    
    dim = len(X[0])

    cov_mat = np.cov(X_meaned , rowvar = False)
    cor_mat = np.corrcoef(X_meaned, rowvar = False)
    init_cov_mat = np.cov(init_mean , rowvar = False)
    init_cor_mat = np.corrcoef(init_mean, rowvar = False)


    cov_eigen_values , _ = np.linalg.eigh(cov_mat)
    cor_eigen_values , _ = np.linalg.eigh(cor_mat)
    init_cov_eigen_values , _ = np.linalg.eigh(init_cov_mat)
    init_cor_eigen_values , _ = np.linalg.eigh(init_cor_mat)
    
    #sort the eigenvalues in descending order
    cov_sorted_index = np.argsort(cov_eigen_values)[::-1]
    cor_sorted_index = np.argsort(cor_eigen_values)[::-1]
    init_cov_sorted_index = np.argsort(init_cov_eigen_values)[::-1]
    init_cor_sorted_index = np.argsort(init_cor_eigen_values)[::-1]
    

    cov_eigen_values = cov_eigen_values[cov_sorted_index]
    cor_eigen_values = cor_eigen_values[cor_sorted_index]
    init_cov_eigen_values = init_cov_eigen_values[init_cov_sorted_index]
    init_cor_eigen_values = init_cor_eigen_values[init_cor_sorted_index]

    cov_eigen_values = cov_eigen_values/np.sum(cov_eigen_values)
    cov_eigen_values = cov_eigen_values.cumsum()
    cor_eigen_values = cor_eigen_values/np.sum(cor_eigen_values)
    cor_eigen_values = cor_eigen_values.cumsum()
    init_cov_eigen_values = init_cov_eigen_values/np.sum(init_cov_eigen_values)
    init_cov_eigen_values = init_cov_eigen_values.cumsum()
    init_cor_eigen_values = init_cor_eigen_values/np.sum(init_cor_eigen_values)
    init_cor_eigen_values = init_cor_eigen_values.cumsum()

    length_cov = np.argwhere(cov_eigen_values>=threshold)[0][0]+1
    length_cor = np.argwhere(cor_eigen_values>=threshold)[0][0]+1
    length_init_cov = np.argwhere(init_cov_eigen_values>=threshold)[0][0]+1
    length_init_cor = np.argwhere(init_cor_eigen_values>=threshold)[0][0]+1

    cov_x = length_cov/dim
    cor_x = length_cor/dim
    cov_x_init = length_init_cov/(dim +1)
    cor_x_init = length_init_cor/(dim +1)
    expl_var_cov_x = cov_eigen_values[0]
    expl_var_cor_x = cor_eigen_values[0]
    expl_var_cov_init = init_cov_eigen_values[0]
    expl_var_cor_init = init_cor_eigen_values[0]

    return cov_x, cor_x, cov_x_init, cor_x_init, expl_var_cov_x, expl_var_cor_x, expl_var_cov_init, expl_var_cor_init