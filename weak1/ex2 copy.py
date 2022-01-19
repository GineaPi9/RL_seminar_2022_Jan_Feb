import gym
import numpy as np
env = gym.make('FrozenLake-v1')
env.render()
print("")

def QVI(env, discount_factor=1.0, theta=0.00001):
    Q = np.zeros([env.nS, env.nA])
    
    while True :
        delta = 0

        for s in range(env.nS):
            q = np.zeros(env.nA)

            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    q[a] += prob*( reward + discount_factor*np.max(Q[next_state]))
                
                delta = max(delta, np.abs(q[a]-Q[s][a]))

            for i in range(env.nA):
                Q[s][i]=q[i]

        
        if delta < theta:
            break

    return np.array(Q)


q = QVI(env)
print("Q value Function:")
print(q)
print("")

pi = np.zeros(env.nS)
for k in range(env.nS):
    pi[k] = np.argmax(q[k])

policy = q.argmax(axis=1)    
print("Pi policy Function:")
print(policy.reshape(4,4))
print("")

# simulation
done = False
nA = env.action_space.n
state = env.reset()
env.render()

while not done:
    action = policy[state]
    new_state, reward, done, info = env.step(action)
    env.render()
    state = new_state

print('\n')
print(pi[state])
print(np.argmax(q[state]))


#while not done :
#    action = pi[state]
#    new_state, reward, done, info = env.step(action)
#    env.render()
#    state = new_state


#while not done:
#    action = np.argmax(q[state])
#    new_state, reward, done, info = env.step(action)
#    env.render()
#    state = new_state


