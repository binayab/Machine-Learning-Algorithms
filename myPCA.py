import numpy as np

def myPCA(X,k):
    #standardize the dataset
    mu = np.mean(X, axis = 0)
    X_std = X - mu
   #X_std = X_std.transpose()

   # PCA with a covariance matrix 
    #covariance and maximum eigenvalue

    covariance_matrix = np.cov(X_std, rowvar = False)
    eig_vals, eig_vecs = np.linalg.eig(covariance_matrix)
    
    sorted_components = np.argsort(eig_vals)[::-1]
    
    sorted_eig_vecs = eig_vecs[sorted_components]
    
    W = sorted_eig_vecs[:,0:k]
    return W,mu

def ProjectDatapoints(X,W,mu):
    X_std = X - mu
   
   # projected data
    X_new = np.dot(W.transpose(), X_std.transpose()).transpose()
    #X_new = X_std.dot(W.transpose())
    
    return X_new
        
