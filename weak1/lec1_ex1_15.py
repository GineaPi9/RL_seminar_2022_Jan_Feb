import numpy as np


rho = [0.8, 0.2] # initial state distribution
P = [[0.58, 0.42], [0.66, 0.34]] # State transition matrix under policy

A = np.identity(2) # identity matrix
R = [-0.02, -0.18] # Expected rewards vector
R = np.transpose(R)
gamma = 0.9

for k in range(0,100):

    A = np.identity(2)+gamma*np.matmul(P,A)
    V = np.matmul(A, R)
    cost = np.matmul(rho, V)
    print('Time step = {},\t Cost = {}'.format(k,cost))
