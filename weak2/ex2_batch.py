import gym
import numpy as np
import sys
from collections import defaultdict

env = gym.make('FrozenLake-v1')
# env = gym.make('FrozenLake-v0', is_slippery=False)
env = gym.wrappers.TimeLimit(env, max_episode_steps = 20)


def generate_episode(policy, env):
    # we initialize the list for storing states, actions, and rewards
    states, actions, rewards = [], [], []
    # Initialize the gym environment
    observation = env.reset()

    while True:
        # append the states to the states list
        states.append(observation)
        # now, we select an action using our sample_policy function and append the action to actions list
        probs = policy[observation]
        action = np.random.choice(np.arange(len(probs)), p=probs)
        actions.append(action)
        # We perform the action in the environment according to our sample_policy, move to the next state
        # and receive reward
        observation, reward, done, info = env.step(action)
        rewards.append(reward)
        # Break if the state is a terminal state
        if done:
            break
    return states, actions, rewards


def first_visit_mc_prediction(env,random_policy, n_episodes):
    # Fill in this part
    V = np.zeros(env.nS)
    memory = defaultdict(list)
    gamma = 1

    for _ in range(n_episodes):
        states, actions, rewards = generate_episode(random_policy, env)
        returns = 0

        for t in range(len(states)-1, -1, -1):
            r = rewards[t]
            a = actions[t]
            s = states[t]
            returns = gamma*returns + r
        
            if s not in states[:t]:
                memory[int(s)].append(returns)
                V[s] = np.average(memory[int(s)])
    
    return V

random_policy = np.ones([env.nS, env.nA]) / env.nA
V = first_visit_mc_prediction(env,random_policy, n_episodes=10000)

print("Value Function:")
print(V.reshape(4,4))    

