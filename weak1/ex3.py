import gym
import numpy as np
env = gym.make('FrozenLake-v1')
# env = gym.make('FrozenLake-v0', is_slippery=False)


def compute_value_function(policy, gamma):
    # [Fill in this part: Value iteratoin]
    theta = 0.00001 # 오차 허용 범위
    V = np.zeros(env.nS) # Value functino(table) 초기화
    while True : 
        delta = 0
        for s in range(env.nS):
            v = 0    

            for transition, next_state, reward, done in env.P[s][policy[s]] :
                v += transition*(reward + gamma*V[next_state])

            delta = max(delta, np.abs(v-V[s]))
            V[s] = v


        if delta < theta: break
    
    print(V.reshape(4,4),"\n")
    return V
  


def extract_policy(value_table, gamma):
    # [Fill in this part: extract a greedy policy for the value]
    pi = np.zeros(env.nS, dtype = int)


    for s in range(env.nS) :      
        v = np.zeros(env.nA)

        for a in range(env.nA): 

            for transition, next_state, reward, done in env.P[s][a]:
                v[a] += transition * (reward + gamma*value_table[next_state])
                #print(v[a])
                #print("state : {} , action : {}".format(s,a))
                #print("")

        
        pi[s] = np.argmax([v[a] for a in range(env.nA)])
        

    return pi




def policy_iteration(env, gamma):
    random_policy = np.zeros(env.observation_space.n)
    no_of_iterations = 200000

    for i in range(no_of_iterations):

        new_value_function = compute_value_function(random_policy, gamma) # policy evaluation
        new_policy = extract_policy(new_value_function, gamma) # policy improvement

        if (np.all(random_policy == new_policy)):
            print ('Policy-Iteration converged at step %d.' %(i+1))
            break
        
        random_policy = new_policy

    return new_policy




optimal_policy = policy_iteration(env,gamma=0.9) 
print(optimal_policy.reshape(4,4))

optimal_policy_int = optimal_policy.astype(int)

# Simulate policy
done = False
state = env.reset()
env.render()
while not done:
    action = optimal_policy_int[state]
    new_state, reward, done, info = env.step(action)
    env.render()
    print("")
    state = new_state

