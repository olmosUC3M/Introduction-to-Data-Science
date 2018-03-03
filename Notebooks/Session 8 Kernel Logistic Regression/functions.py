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

#Construct the function that computes de cost and its gradient
def cost(a,N,K,Y,l):
    
    
    grad=np.zeros([N,1])
    
    h = sigmoid(K@a)
    
    L = -1.0*(np.transpose(Y)@np.log(h)+np.transpose(1.0-Y)@np.log(1.0-h))+l*np.transpose(a)@(K@a)
    
    L /= float(N)
    
    grad += np.sum(np.tile(np.transpose(h-Y),[N,1])*K,axis=1).reshape(N,1)
    
    grad /= (float(N))
    
    grad += 2.0*l*(K@a)/(float(N))


    return L,grad

def optimize_KLR(N,K,Y,l,a_init,max_iter,step,tolerance,verbose=True):

    a=a_init
    convergence=False
    it=0
    while convergence==False:
        L,grad=cost(a,N,K,Y,l)
        a=a-step*grad
        
        if(np.linalg.norm(grad)<tolerance): 
            convergence=True
        
        if(it>=max_iter):
            convergence=True
        
        if(verbose==True and it%100==0):
            print('Iteration %d, cost L=%f, grad_norm =%f' %(it,L,np.linalg.norm(grad)))
            
        it=it+1
    
    print('Iteration %d, cost L=%f, grad_norm =%f' %(it,L,np.linalg.norm(grad)))
    return a

def compute_Kernel_matrix(X,N,bandwidth,kernel='rbf'):
    
    K=np.zeros([N,N])
    
    if(kernel=='rbf'):
        for i in range(0,N):
            for j in range (0,N):
                K[i,j]=np.exp(-1*np.linalg.norm(X[i,:]-X[j,:])**2/(2*bandwidth))
    else:
        print('Kernel function not implemented')
        
    return K


def compute_kernel_vector(X,v,N,bandwidth,kernel='rbf'):
    
    k = np.zeros([N,1])

    if(kernel=='rbf'):
        for i in range(0,N):            
            k[i] = np.exp(-1*np.linalg.norm(X[i,:]-v)**2/(2*bandwidth))    
    else:
        print('Kernel function not implemented')
        
    return k


def sigmoid(s):
    return 1.0/(1.0+np.exp(-1.0*s))