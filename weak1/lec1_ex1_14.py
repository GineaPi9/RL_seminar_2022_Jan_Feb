import numpy as np


rho = [0.4, 0.6] # initial state distribution
P = [[0.58, 0.42], [0.66, 0.34]]

d = rho

for k in range(0,20):

    d = np.matmul(d, P)
    print('time step = {},\t state distribution = {}'.format(k,d))