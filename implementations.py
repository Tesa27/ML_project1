#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def ridge_regression(y, tx, lambda_):
    #Ridge regression using normal equations
    # y:          vector of outputs (dimension N)
    # tx:         matrix of data (dimension N x D), such that tx[:, 0] = 1
    #lambda_:     regularization parameter
    N,D = tx.shape
    
    A = np.dot(tx.T, tx) + lambda_ * np.ones(D)
    B = np.linalg.inv(A)
    w = np.dot(np.dot(B,tx.T), y)
    
    # Calculating loss
    r = y - np.dot(tx,w)
    loss = (np.dot(r,r)+ lambda_ * np.dot(w,w)) / (2*N)
    
    return w, loss
def least_squares_GD(y, tx, initial_w,max_iters, gamma):
    """
    Linear regression using gradient descent and least squares
    """
    N, D = tx.shape
    
    # Iterations of gradient descent
    w = initial_w
    for _ in range(max_iters):
        grad = -np.dot(tx.T, (y - np.dot(tx,w))) / (N)
        w = w - gamma * grad
        
    # Calculating the loss
    r = y - np.dot(tx,w)
    loss = np.dot(r,r) / (2*N)
    
    return w, loss

#Linear regression using stochastic gradient descent
def least_squares_SGD(y, tx, initial_w,max_iters, gamma, frequency=0):
    """Linear regression using stochastic gradient descent and least squares"""
    N, D = tx.shape
    
    # Iterations of stochastic gradient descent
    w = initial_w
    for i in range(max_iters):
        k = np.random.randint(0,N-1)
        grad = -(y[k]-np.dot(tx[k,:], w))*tx[k,:]
        w = w - gamma * grad
        
            
    r = y - np.dot(tx,w)
    loss = np.dot(r,r) / (2*N)    
    
    return w, loss

#Least squares regression using normal equations
def least_squares(y, tx):
    N, _ = tx.shape
    
    # Calculating w
    w = (np.linalg.inv((tx.T).dot(tx)).dot(tx.T)).dot(y)
    
    #Calculating loss
    r = y - tx.dot(w)
    loss = np.dot(r,r)/(2*N)
    return w, loss

def logistic_regression(y, tx, initial_w,max_iters, gamma):
    """
    #Logistic regression using SGD
    # y:          vector of outputs (dimension N)
    # tx:         matrix of data (dimension N x D)
    # initial_w:  vector (dimension D)
    # max_iters:  scalar
    # gamma:      scalar respresenting step size
    # return parameters w for the regression and loss
    """
    return reg_logistic_regression(y, tx, 0, initial_w,max_iters, gamma)

def reg_logistic_regression(y, tx, lambda_ ,initial_w, max_iters, gamma):
    """
    #Regularized logistic regression using SGD
    # y:          vector of outputs (dimension N)
    # tx:         matrix of data (dimension N x D), such that tx[:, 0] = 1
    # lambda:     scalar representing regularization parameter
    # initial_w:  vector (dimension D)
    # max_iters:  scalar
    # gamma:      scalar respresenting step size
    # return parameters w for the regression and loss
    """
    N, _ = tx.shape
    w = initial_w
    
    
    for i in range(max_iters):
        k = np.random.randint(0,N-1)
        tmp = np.dot(tx[k,:],w)
        grad = -y[k]*tx[k,:]+sigmoid(tmp)*tx[k,:]+lambda_*w
        w = np.squeeze(np.asarray(w - gamma*grad))
        
        
        
    tmp = np.squeeze(np.asarray(np.dot(tx,w)))
    loss = - np.dot(tmp, y.T)
    loss += np.sum(np.log(1+np.exp(tmp)))
    loss /= (2*N)

    return w, loss

