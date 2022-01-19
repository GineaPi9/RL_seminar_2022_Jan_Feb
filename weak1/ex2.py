import gym
import numpy as np
env = gym.make('FrozenLake-v1')
#env = gym.make('FrozenLake-v1', is_slippery=False)
env.render()
print("")

def QVI(env, discount_factor=1.0, theta=0.00001):
    Q = np.zeros([env.nS, env.nA])

    while True:
        delta = 0

        # For each state, perform a "full backup"
        for s in range(env.nS): 
            q = np.zeros(env.nA) 
            for a in range(env.nA):
                # state s, action a -> Optimal Q value 계산  
                for  prob, next_state, reward, done in env.P[s][a]: 
                    # [Fill in this this part: Q-value iteration part]
                    q[a] += prob*(reward + discount_factor*Q[next_state].max())

                delta = max(delta, np.abs(q[a] - Q[s][a]))


            #for i in range(env.nA):
            #    Q[s][i] = q[i]
            Q[s] = q # 위 두 줄이랑 똑같은 동작

        if delta < theta: 
            break

    return np.array(Q)


q = QVI(env)
print("Q value Function:")
print(q)
print("")

# Extract optimal policy 1
pi = np.zeros(env.nS)
for s in range(env.nS):
    pi[s] = np.argmax(q[s])
print("Pi policy Function:")
print(pi.reshape(4,4))
print("")

# Extract optimal policy 2
opt_policy = q.argmax(axis=1)    
print("Pi policy Function:")
print(opt_policy.reshape(4,4))
print("")

# Simulate the optimal policy
done = False
nA = env.action_space.n
state = env.reset()
env.render()

i =0
while not done:
    action = np.argmax(q[state]) # action = opt_policy[state]
    new_state, reward, done, info = env.step(action)
    env.render()
    state = new_state

    i += 1
    print(i)


#def q_value_iteration(env, q_table, discount_factor=1.):
#    next_q_table = np.zeros((env.nS, env.nA))
#    for s in range(env.nS):
#        for a in range(env.nA):
#            for prob, next_state, reward, done in env.P[s][a]:
#                next_q_table[s, a] += prob * (reward + (1-done) * discount_factor * q_table[next_state].max())
#    return next_q_table

#def main():
#    env = gym.make('FrozenLake-v0', is_slippery=True)
#    q_table = np.zeros((env.nS, env.nA))
    
#    for i in range(500):
#        q_table = q_value_iteration(env, q_table=q_table)
#    policy = q_table.argmax(axis=1)
    
#    print(q_table)
#    print(policy.reshape(4, 4))
#main()