import gym
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
from time import sleep
import matplotlib.pyplot as plt
from collections import deque


#Hyperparameters
lr_mu        = 0.001 # 0.005
lr_q         = 0.001
gamma        = 0.99
batch_size   = 63
buffer_limit = 50000
tau          = 0.005 # for target network soft update

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0 
            done_mask_lst.append([done_mask])
        
        return torch.tensor(np.array(s_lst), dtype=torch.float), torch.tensor(np.array(a_lst), dtype=torch.float), \
                torch.tensor(r_lst, dtype=torch.float), torch.tensor(np.array(s_prime_lst), dtype=torch.float), \
                torch.tensor(done_mask_lst, dtype=torch.float)
    
    def size(self):
        return len(self.buffer)

class MuNet(nn.Module):
    def __init__(self):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(24, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x)) # Multipled by 2 because the action space of the Pendulum-v0 is [-2,2]
        return mu

class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(24, 64)
        self.fc_a = nn.Linear(4,64)
        self.fc_q = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64,1)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1,h2], dim=1)
        q = F.relu(self.fc_q(cat))
        q = self.fc_out(q)
        return q

    
class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x
      




def train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer):
    #Fill this part

    mini_batch = memory.sample(batch_size)
    s_lst, a_lst, r_lst, next_s_lst, d_mask_lst = mini_batch[0], mini_batch[1], mini_batch[2], mini_batch[3], mini_batch[4]

    # update Q param
    Q_current = q(s_lst,a_lst)
    next_a = mu_target(next_s_lst) # next action (Using target param)
    Q_prime = q_target(next_s_lst, next_a) # Next Q value (Using target param)
    target = r_lst + gamma*Q_prime*d_mask_lst # Target value 
    Q_loss = 0.5*F.mse_loss(target, Q_current, reduction='mean') # Q value loss

    q_optimizer.zero_grad()
    Q_loss.backward()
    q_optimizer.step()


    # update policy param
    p_loss = 0
    actions = mu(s_lst) # current deterministic action
    Q_w = q(s_lst, actions)
    P_loss = -torch.mean(Q_w)
   

    mu_optimizer.zero_grad()
    P_loss.backward()
    mu_optimizer.step()

    
def soft_update(net, net_target): # target parameter update
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
    
env = gym.make('BipedalWalker-v3')
memory = ReplayBuffer()

q, q_target = QNet(), QNet() # initialize Q_function & parameter, Q_target & parameters
q_target.load_state_dict(q.state_dict())  # initialize Q_target 
mu, mu_target = MuNet(), MuNet()  # initialize policy & parameter, p_target & paramer
mu_target.load_state_dict(mu.state_dict()) # # initialize p_target 

score = 0.0 
print_interval = 20
reward_history =[] 
reward_history_100 = deque(maxlen=100)


mu_optimizer = optim.Adam(mu.parameters(), lr=lr_mu)
q_optimizer  = optim.Adam(q.parameters(), lr=lr_q) #
ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1)) # Noise 

MAX_EPISODES = 500
for episode in range(MAX_EPISODES):
    s = env.reset() # observe s_0
    done = False
        
    while not done: # generate episode with deterministic policy 
        a = mu(torch.from_numpy(s).float()) # get deterministic action
        a = a.detach().numpy() + ou_noise()[0] # deterministic action + noise
        s_prime, r, done, info = env.step(a)
        memory.put((s,a,r/100.0,s_prime,done)) # preserve sample in memory(Buffer)
        score = score + r # update score for current episode
        s = s_prime # move to next state.
                
    if memory.size()>2000:
        for i in range(10):
            train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer)
            soft_update(mu, mu_target)
            soft_update(q,  q_target)
        
    reward_history.append(score)
    reward_history_100.append(score)
    avg = sum(reward_history_100) / len(reward_history_100)
    episode = episode + 1
    if episode % print_interval == 0:
        print('episode: {}, epi. score: {:.1f}, avg(100 epi.): {:.1f}'.format(episode, score, avg))
    score = 0.0

env.close()




from matplotlib import pyplot as plt
import numpy as np
## Plot objective vs. iteration
t = range(MAX_EPISODES)
plt.plot(t, np.array(reward_history), 'b', linewidth = 2, label = 'DDPG')
plt.legend(prop={'size':12})
plt.xlabel('Episode')
plt.ylabel('Total rewards')
plt.show()



# # Test
# from time import sleep
# episode = 0
# s = env.reset()   
# while episode < 5:  # episode loop
#     env.render()
#     a = mu(torch.from_numpy(s).float()) 
#     a = a.detach().numpy()
#     s_prime, r, done, info = env.step([a])
#     s = s_prime
#     sleep(0.01)
    
#     if done:
#         env.close()               
#     score = score + r    
        
#     if done:
#         episode = episode + 1
#         print('Episode: {} Score: {}'.format(episode, score))
#         state = env.reset()
#         score = 0
        

# env.close()