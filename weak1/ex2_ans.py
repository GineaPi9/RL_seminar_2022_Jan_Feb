import gym
import numpy as np

def q_value_iteration(env, q_table, discount_factor=1.):
    next_q_table = np.zeros((env.nS, env.nA))
    for s in range(env.nS):
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[s][a]:
                next_q_table[s, a] += prob * (reward + (1-done) * discount_factor * q_table[next_state].max())

    return next_q_table


def main():
    env = gym.make('FrozenLake-v1', is_slippery=True)
    q_table = np.zeros((env.nS, env.nA))
    
    for i in range(1000):
        q_table = q_value_iteration(env, q_table=q_table)
    policy = q_table.argmax(axis=1)
    
    print(q_table,"\n")
    #print(policy.reshape(4, 4))

    #Extract optimal policy
    pi = np.zeros(env.nS)
    
    for s in range(env.nS):
        pi[s] = np.argmax(q_table[s])
    
    print("Pi policy Function:")
    print(pi.reshape(4,4))
    print("")

    # Simulate the optimal policy
    done = False
    nA = env.action_space.n
    state = env.reset()
    env.render()

    while not done:
        action = np.argmax(q_table[state])
        new_state, reward, done, info = env.step(action)
        env.render()
        state = new_state

main()

