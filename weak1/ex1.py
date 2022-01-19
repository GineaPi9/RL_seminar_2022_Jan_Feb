import gym
import numpy as np

env = gym.make('FrozenLake-v1')
#env = gym.make('FrozenLake-v1', is_slippery=False)

def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    # Start with a random (all 0) value function
    V = np.zeros(env.nS) # initial Value

    while True:
        delta = 0
        
        # For each state, perform a "full backup"
        for s in range(env.nS): # env.nS = 16
            v = 0
            # Look at the possible next actions
            for a, action_prob in enumerate(policy[s]):
                # For each action, look at the possible next states...    
                for  transition_prob, next_state, reward, done in env.P[s][a]:                  
                    # [Fill in this part]
                    v += action_prob*transition_prob*(reward+discount_factor*V[next_state])
                    # print(env.P[s][a])

            # How much our value function changed (across any states)
            delta = max(delta, np.abs(v - V[s])) # 
            V[s] = v
            #print(V)
            #print("")


        # Stop evaluating once our value function change is below a threshold
        if delta < theta:
            break

    return np.array(V)



random_policy = np.ones([env.nS, env.nA]) / env.nA # Uniform random policy
v = policy_eval(random_policy, env) # policy evaluation


print("Value Function(DP - Policy Evaluation):")
print(v.reshape(4,4))
print("")



# 전체 152 번