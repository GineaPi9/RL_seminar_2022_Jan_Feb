import numpy as np
import math

P1 = [[0.5, 0.5], [0.9, 0.1]] # State transition matrix under action 1
P2 = [[0.6, 0.4], [0.3, 0.7]] # State transition matrix under action 2
R1 = [1.5, -0.1] # Expected rewards vector under action 1
R2 = [-0.4, -0.3] # Expected rewards vector under action 2
R1 = np.transpose(R1)
R2 = np.transpose(R2)
gamma = 0.9


V = [0,0]
V = np.transpose(V)

for k in range(0,100):

    a1 = np.max([R1[0]+gamma*np.matmul(P1[0],V) , R2[(0)]+gamma*np.matmul(P2[0],V)])
    a2 = np.max([R1[1]+gamma*np.matmul(P1[1],V) , R2[(1)]+gamma*np.matmul(P2[1],V)])
    V = np.transpose([a1, a2])
    print('Time step = {},   V={}'.format(k,V))


print(P1[(0)])