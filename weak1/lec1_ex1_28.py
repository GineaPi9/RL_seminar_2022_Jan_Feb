import numpy as np
import math

P = [[0.58, 0.42], [0.66, 0.34]] # State transition matrix under policy
R = [-0.02, -0.18] # Expected rewards vector
R = np.transpose(R)
gamma = 0.9

V = [0,0] # initialized value 
V = np.transpose(V)

for k in range(0,100):

    V = R + gamma*np.matmul(P,V)

    print( "Time step = {}, \t V= {}".format(k,np.round(V,5)) )
