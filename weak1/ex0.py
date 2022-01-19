import gym
from time import sleep

env = gym.make('FrozenLake-v1') # Slippery -> action random
#env = gym.make('FrozenLake-v1', is_slippery = False) # 미끄럼 방지 - action random 아님

print("Action Space {}".format(env.action_space)) # Action 개수

# continuous state-space
print("State Space {}".format(env.observation_space)) # State 개수
print(env.observation_space.n)
print(env.action_space.n)
env = gym.wrappers.TimeLimit(env, max_episode_steps = 10) # 최대 step 제한
env.reset() # 초기 state 반환/이동



s=9
a=1
print(env.P[s][a]) # 

prob1, prob2, prob3 = env.P[0][0]

transition1 = prob1[0] 
print(transition1)


done = False
env.render()

while not done:
    new_state, reward, done, info = env.step(env.action_space.sample())
    env.render()
    print(reward, new_state)
    sleep(.1)

# for state in range(env.observation_space.n):
#    for action in range(env.action_space.n):

# print(env.P[1][1])