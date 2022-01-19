import gym
import matplotlib
import numpy as np
import random

env = gym.make('FrozenLake-v1')
# env = gym.make('FrozenLake-v0', is_slippery=False)
env = gym.wrappers.TimeLimit(env, max_episode_steps = 50)

def epsilon_greedy(state,Q, epsilon):
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action

def generate_episode(Q, epsilon, env):
    states, actions, rewards = [], [], []
    state = env.reset()
    while True:
        states.append(state)
        action = epsilon_greedy(state,Q, epsilon)
        actions.append(action)
        state, reward, done, info = env.step(action)
        rewards.append(reward)
        if done:
             break
    
    #print(actions)
    return states, actions, rewards


#  Now that we learned how to generate an episode, we will see how to perform First Vist MC Prediction

def first_visit_mc_prediction(env, epsilon, n_episodes):
    # Fill in this part
    Q = np.zeros((env.nS, env.nA))
    m = np.zeros((env.nS,env.nA))
    gamma = 1

    for _ in range(n_episodes):
        states, actions, rewards = generate_episode(Q, epsilon, env)
        returns = 0 

        for t in range(len(states)-1,-1,-1):
            r = rewards[t]
            a = actions[t]
            s = states[t]
            returns = returns*gamma + r

            if s not in states[:t] or a not in states[:t] :
                m[s,a] = m[s,a] + 1
                Q[s,a] = Q[s,a] + (1/m[s,a])*(returns - Q[s,a])


    return Q

Q = first_visit_mc_prediction(env,0.2, n_episodes=50000)

print(Q,'\n')

policy = Q.argmax(axis=1)    
print("Pi policy Function:")
print(policy.reshape(4,4))
print("")


done = False
state = env.reset()
env.render()
while not done:
    action = np.argmax(Q[state])
    new_state, reward, done, info = env.step(action)
    env.render()
    state = new_state


