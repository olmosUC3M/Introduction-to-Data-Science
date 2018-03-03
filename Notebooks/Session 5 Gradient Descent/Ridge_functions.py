import numpy as np


# A function that normalizes data with pre-scecified mean and std. 
def normalize(X,mu=0,std=1,flag_train=True):
    if(flag_train):
        mu = np.mean(X,0)
        std = np.std(X,0)
    
    X -= mu
    X /= std
    
    return X,mu,std

# A function to add the all-ones column
def add_interfit(X):
    col_ones = np.ones([X.shape[0],1])
    return np.concatenate([col_ones,X],1)

# A function to create the normalized function with polynomial features up to degree deg_max
def create_feature_matrix(X_0,deg_max,mu=0,std=1,flag_train=True):
    X = np.zeros([X_0.shape[0],deg_max])
    
    deg=1
    while deg<=deg_max:
        X[:,deg-1] = X_0**deg 
        deg += 1
    
    X,train_mean,train_std = normalize(X,mu,std,flag_train)
    
    X = add_interfit(X)
    
    return X,train_mean,train_std

# A function to evaluate the LS solution
def LS_evaluate(X,T):
    return (X @ T.transpose())

# A function that calculates the error
def J_error(Y,Y_est):
    return np.mean((Y-Y_est)**2)

# A function that calculates the error + L2 penalization
def J_error_L2(Y,Y_est,T,l):
    return  J_error(Y,Y_est) + (l/Y.shape[0]) * np.sum(T**2)

# A function to compute the LS solution
def Ridge_solution(X,Y,l):
    A = l*np.eye(X.shape[1])
    A[0,0] = 0
    A += X.transpose() @ X 

    return (np.linalg.inv(A) @ X.transpose() @ Y)        

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

# A function to compute the Ridge cost function for both train and validation/test sets using 
# a provided value of the parameter vector T, polynomial degree and lambda_value

def eval_J_Ridge_given_T(X_train,Xvt,degree,Y_train,Yvt,l,T):
    
    # Xvt,Yvt --> We use this function to evaluate either validation error or test error
    
    # Lets compute the normalized feature matrices F_train, F_test
    F_train,train_mean,train_std = create_feature_matrix(X_train,degree,0,1,flag_train=True)
    
    F_vt,_,_ = create_feature_matrix(Xvt,degree,train_mean,train_std,flag_train=False)
    
    
    # We evaluate the Penalized MSE (MSE + L2 penalization)
    J_train = J_error_L2(Y_train,LS_evaluate(F_train,T),T,l)  
    J_vt = J_error_L2(Yvt,LS_evaluate(F_vt,T),T,l)        


        
    return J_train,J_vt,F_train,F_vt