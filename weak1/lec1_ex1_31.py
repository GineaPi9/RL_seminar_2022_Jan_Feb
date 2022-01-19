import numpy as np
import math

P1 = [[0.5, 0.5], [0.9, 0.1]] # State transition matrix under action 1
P2 = [[0.6, 0.4], [0.3, 0.7]] # State transition matrix under action 2
R1 = [1.5, -0.1] # Expected rewards vector under action 1
R2 = [-0.4, -0.3] # Expected rewards vector under action 2
R1 = np.transpose(R1)
R2 = np.transpose(R2)
gamma = 0.9


Q1 = np.transpose([0,0])
Q2 = np.transpose([0,0])


for k in range(0,100):

    a1 =  R1[0] + gamma*(P1[0][0]*np.max([Q1[0],Q2[0]]) + P1[0][1]*np.max([Q1[1],Q2[1]]))
    a2 =  R2[0] + gamma*(P2[0][0]*np.max([Q1[0],Q2[0]]) + P2[0][1]*np.max([Q1[1],Q2[1]]))
    a3 =  R1[1] + gamma*(P1[1][0]*np.max([Q1[0],Q2[0]]) + P1[1][1]*np.max([Q1[1],Q2[1]]))
    a4 =  R2[1] + gamma*(P2[1][0]*np.max([Q1[0],Q2[0]]) + P2[1][1]*np.max([Q1[1],Q2[1]]))
    
    Q1 = np.transpose([a1, a3])
    Q2 = np.transpose([a2, a3])
    Q = [a1, a2, a3, a4]
    print('Time step = {}, \t Q = {}'.format(k,Q))
