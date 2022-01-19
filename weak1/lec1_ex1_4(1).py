import numpy as np

rho = [0.5, 0.5]
P11 = [0.5, 0.5]
P21 = [0.9, 0.1]
P12 = [0.6, 0.4]
P22 = [0.3, 0.7]

S = ["State 1", "State 2"]

current_state = np.random.choice(S, 1, p=rho) # S 중에서 1개를 p=rho 확률로 선택
action_sequence = [1, 2, 1, 2, 2, 1, 2, 1, 1, 1] # action 순서

for k in range(0, 10):
    current_action = action_sequence[k]

    if current_state == "State 1" and current_action == 1:
        next_state = np.random.choice(S,1,p=P11)

    elif current_state == "State 1" and current_action == 2:
        next_state = np.random.choice(S,1,p=P12)

    elif current_state == "State 2" and current_action == 1:
        next_state = np.random.choice(S,1,p=P21)

    elif current_state == "State 2" and current_action == 2:
        next_state = np.random.choice(S,1,p=P22)

    print(str(current_state), current_action)
    current_state = next_state