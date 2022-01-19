import gym
import numpy as np
from collections import defaultdict


env = gym.make('FrozenLake-v1')
# env = gym.make('FrozenLake-v0', is_slippery=False)
env = gym.wrappers.TimeLimit(env, max_episode_steps = 20)

# We define a function called generate_episode for generating epsiodes
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


#  Now that we learned how to generate an episode, we will see how to perform First Vist MC Prediction
def every_visit_mc_prediction(env, random_policy, n_episodes):
    # Fill in this part
    Q = np.zeros([env.nS, env.nA])
    gamma = 1
    memory = defaultdict(list)

    for _ in range(n_episodes):
        states, actions, rewards = generate_episode(random_policy,env)
        returns = 0

        for t in range(len(states)-1, -1, -1):
            r = rewards[t]
            s = states[t]
            a = actions[t]
            returns = gamma*returns + r
            memory[int(s),int(a)].append(returns)
            # V[s] = np.average(memory[int(s)])
            Q[s,a] = np.average(memory[int(s),int(a)])


    return Q

random_policy = np.ones([env.nS, env.nA]) / env.nA
Q = every_visit_mc_prediction(env,random_policy, n_episodes = 10000)

V = np.zeros(env.nS)
for s in range(env.nS):
    for a in range(env.nA):
        V[s] = V[s] + random_policy[s,a] * Q[s,a]
print("Value Function:")
print(V.reshape(4,4))    




