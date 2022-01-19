import numpy as np

rho = [0.5, 0.5]


S = ["State 1", "State 2"]

def pi(state):

    if state == "State 1":
        action = 2
    elif state == "State 2":
        action = 1
    return action

def P(state, action):
    if state == "State 1" and action == 1:
        p = [0.5, 0.5]
    elif state == "State 2" and action == 1:
        p = [0.9, 0.1]
    elif state == "State 1" and action == 2:
        p = [0.6, 0.4]
    elif state == "State 2" and action == 2:
        p = [0.3, 0.7]
    return p

current_state = np.random.choice(S,1,p=rho)

for k in range(0, 10):
    current_action = pi(current_state) # 현재 current 에 따라 current action 선택
    next_state = np.random.choice(S, 1, p=P(current_state, current_action))

    print(str(current_state),current_action)
    current_state = next_state
