import numpy as np

P = [[0.1, 0.4, 0.3, 0.2],[0.12, 0.48, 0.24, 0.16],[0.18,0.72,0.06,0.04],[0.06,0.24,0.42,0.28]] # State transition matrix under policy
R = [1.5, -0.4, -0.1, -0.3] # Expected rewards vector
R = np.transpose(R)
gamma = 0.9

A = np.identity(4) # identity matrix


for k in range(0,100):

    A = np.identity(4)+gamma*np.matmul(P,A)
    Q = np.matmul(A, R)
    print('Time step = {},\t V = {}'.format(k,Q))
