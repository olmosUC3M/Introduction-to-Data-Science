import numpy as np


# A function that normalizes data with pre-scecified mean and std. 
def normalize(X,mu=0,std=1,flag_train=True):
    X_0 = np.copy(X)
    if(flag_train):
        mu = np.mean(X,0)
        std = np.std(X,0)
    
    X_0 -= mu
    X_0 /= std
    
    return X_0,mu,std

# A function to add the all-ones column
def add_interfit(X):
    col_ones = np.ones([X.shape[0],1])
    return np.concatenate([col_ones,X],1)

# A function to randomly split a data set
def split_set(X_0,Y_0,fraction):
    
    N = X_0.shape[0]
    N_split = np.round(fraction * X_0.shape[0]).astype(np.int32)
    mask = np.random.permutation(N)
    
    
    X_1 = X_0[mask[N_split:-1]]
    Y_1 = Y_0[mask[N_split:-1]]
    
    X_0 = X_0[mask[:N_split]]
    Y_0 = Y_0[mask[:N_split]]
    
    return X_0,X_1,Y_0,Y_1
