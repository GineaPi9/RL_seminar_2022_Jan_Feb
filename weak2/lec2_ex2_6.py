import numpy as np
from collections import defaultdict

rho = [0.4, 0.4, 0.2]
S = [0,1,2]
A = [0,1]
gamma = 0.9

def pi(state) : 
    if state == 0:
        p = [0.2, 0.8]
    elif state == 1:
        p = [0.6, 0.4]
    elif state == 2:
        p = [0.5, 0.5]    
    return p

def P(state, action):
    if state == 0 and action == 0:
        p = [0.5, 0.4, 0.1]
    elif state == 1 and action == 0:
        p = [0.8, 0.1, 0.1]
    elif state == 2 and action == 0:
        p = [0, 0, 1]
    elif state == 0 and action == 1:
        p = [0.5, 0.4, 0.1]
    elif state == 1 and action == 1:
        p = [0.3, 0.6, 0.1]
    elif state == 2 and action == 1:
        p = [0, 0, 1]
    return p

def reward(state, action, next_state):
    if state == 0 and action == 0 and next_state == 0:
        R = 1
    elif state == 0 and action == 0  and next_state == 1 :
        R = 2
    elif state == 0 and action == 0 and next_state == 2 :
        R = 0
    elif state == 1 and action == 0 and next_state == 0 :
        R = 0
    elif state == 1 and action == 0 and next_state == 1 :
        R = 2
    elif state == 1 and action == 0 and next_state == 2 :
        R = 1
    elif state == 0 and action == 1 and next_state == 0:
        R = 0
    elif state == 0 and action == 1 and next_state == 1:
        R = -1
    elif state == 0 and action == 1 and next_state == 2:
        R = 0
    elif state == 1 and action == 1 and next_state == 0 :
        R = 2
    elif state == 1 and action == 1 and next_state == 1 :
        R = 0
    elif state == 1 and action == 1 and next_state == 2 :
        R = 2
    elif state == 2 :
        R = 1
    return R

def step(state, action) : 
    done = 0
    next_state = np.random.choice(S, 1, p=P(state,action))
    r = reward(state, action, next_state)
    if state == 2:
        done = 1
    return next_state, done, r
       

def reset():
    state = np.random.choice(S, 1, p = rho)
    return state


def generate_episode():
    states, actions, rewards = [], [], []
    state = reset()

    while True : 
        states.append(state)
        action = np.random.choice(A,1,p=pi(state))
        actions.append(action)

        next_state, done, r = step(state, action)
        rewards.append(r)

        if done == 1:
            break
        state = next_state
    
    return states, actions, rewards


def every_visit_mc_prediction(n_episodes) : 
    Q = np.zeros([3,2])
    memory = defaultdict(list)
    for _ in range(n_episodes):
        states, actions, rewards = generate_episode()
        returns = 0

        for t in range(len(states) -1, -1, -1) : # t = 2 -> 1 -> 0
            r = rewards[t]
            s = states[t]
            a = actions[t]
            returns = gamma*returns + r
            memory[int(s), int(a)].append(returns)
            Q[s,a] = np.average(memory[int(s),int(a)])

    return Q

Q = every_visit_mc_prediction(n_episodes = 5000)

V = np.zeros(3)
for s in range(3):
    for a in range(2) : 
        V[s] = V[s] + pi(s)[a]*Q[s,a]
print("Value Function : ")
print(V)