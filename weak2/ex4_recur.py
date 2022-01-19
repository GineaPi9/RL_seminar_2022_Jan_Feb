import gym
import numpy as np
import sys
from collections import defaultdict
env = gym.make('FrozenLake-v1')
# env = gym.make('FrozenLake-v0', is_slippery=False)
env = gym.wrappers.TimeLimit(env, max_episode_steps = 20)

def generate_episode(policy, env):
    states, actions, rewards = [], [], []
    observation = env.reset()
    while True:
        states.append(observation)
        probs = policy[observation]
        action = np.random.choice(np.arange(len(probs)), p=probs)
        actions.append(action)
        observation, reward, done, info = env.step(action)
        rewards.append(reward)
        if done:
             break
    return states, actions, rewards


def first_visit_mc_prediction(env,random_policy, n_episodes):
    # Fill in this part
    Q = np.zeros((env.nS, env.nA))
    m = np.zeros((env.nS,env.nA))
    gamma = 1

    for _ in range(n_episodes):
        states, actions, rewards = generate_episode(random_policy, env)
        returns = 0

        for t in range(len(states)-1, -1, -1):
            r = rewards[t]
            s = states[t]
            a = actions[t]
            returns = gamma*returns + r

            if s not in states[0:t] or a not in actions[0:t] : # (S && A)^c
                m[s,a] = m[s,a] + 1
                Q[s,a] = Q[s,a] + (1/m[s,a])*(returns - Q[s,a])

    return Q


random_policy = np.ones([env.nS, env.nA]) / env.nA
Q = first_visit_mc_prediction(env,random_policy, n_episodes=10000)
print(Q.reshape(env.nS, env.nA), '\n')

V = np.zeros(env.nS)
for s in range(env.nS):
    for a in range(env.nA):
        V[s] = V[s] + random_policy[s,a]*Q[s,a]

print("Value Function:")
print(V.reshape(4,4))    

