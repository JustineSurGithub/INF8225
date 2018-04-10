import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import gym # Game environment
from torch.autograd import Variable
import random
from collections import namedtuple

from random_process import OrnsteinUhlenbeckProcess

USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

class Actor(nn.Module): # policy mu
    def __init__(self, nb_states, nb_actions):
        super(Actor, self).__init__() 
        self.fc1 = nn.Linear(nb_states, 400) # inputs -> nb of features to discribe a state
        self.fc2 = nn.Linear(400, 300) 
        self.fc3 = nn.Linear(300, nb_actions) # outputs -> nb of features to discribe an action
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.tanh(x)
        return x

class Critic(nn.Module): # Q network
    def __init__(self, nb_states, nb_actions):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(nb_states, 400) 
        self.fc2 = nn.Linear(nb_actions+400, 300) 
        self.fc3 = nn.Linear(300, 1) 
    
    def forward(self, state, action):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(torch.cat([x,action],1)))
        q = self.fc3(x)
        return q 



def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
)
        
def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)



class Agent():
    def __init__(self, nb_states, nb_actions):
        self.critic = Critic(nb_states, nb_actions) # Q
        self.critic_target = Critic(nb_states, nb_actions)
        self.actor = Actor(nb_states, nb_actions) # policy mu
        self.actor_target = Actor(nb_states, nb_actions)
        
        hard_update(self.critic_target, self.critic)
        hard_update(self.actor_target, self.actor)
        
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.001)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.0001)
        
        self.criterion = nn.MSELoss()
        
        self.random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=0.15, mu=0, sigma=0.2)

        self.gamma = 0.99
        self.batch_size = 64

        
        if USE_CUDA: 
            self.actor.cuda()
            self.actor_target.cuda()
            self.critic.cuda()
            self.critic_target.cuda()
    
    def act(self, obs, epsilon=0.1): # epsilon -> tunning paramter
        if (random.random() < epsilon): # choose random action
            action = np.random.uniform(-1.,1.,nb_actions)
            return action
        else : # the action is the output of actor network + Exploration Noise
            action = self.actor(obs).cpu().data.numpy()
            action += self.random_process.sample()
            action = np.clip(action, -1., 1.) # to stay in interval [-1,1]
            return action
    
    def backward(self, transitions):       
        
        transitions = memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        
        state_batch = Variable(torch.cat(batch.state)).type(FLOAT) # size 64 x 3
        action_batch = Variable(torch.cat(batch.action)).type(FLOAT) # size 64
        next_state_batch = Variable(torch.cat(batch.next_state)).type(FLOAT) # size 64 x 3
        reward_batch = Variable(torch.cat(batch.reward)).type(FLOAT) # size 64
        done_batch = Variable(torch.cat(batch.done)).type(FLOAT)
        
        
        #### Q - CRITIC UPDATE ####
        
        # Q(s_t,a_t)
        action_batch.unsqueeze_(1) # size 64x1
        state_action_value = self.critic(state_batch,action_batch) # 64x1
        
        # a_{t+1} = mu_target(s_{t+1})
        next_action = self.actor_target(next_state_batch).detach() # 64 x nb_actions

        
        
        # Q'(s_{t+1},a_{t+1})
        next_state_action_value = self.critic_target(next_state_batch, next_action).detach()
        next_state_action_value.squeeze_() # 64
    
        # mask to consider next_state_values to 0 if state is terminal
        mask = Variable(np.logical_not(done_batch.data).type(torch.FloatTensor)).type(FLOAT)
        # mask = 1,1,1 ..
        
        # Compare Q(s_t,a_t) with r_t + gamma * Q'(s_{t+1},a_{t+1})
        expected_state_action_value = reward_batch + (self.gamma * next_state_action_value * mask)
        # Compute Huber loss
        # loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        loss = self.criterion(state_action_value, expected_state_action_value)
        

        # Optimize the nn by updating weights with adam descent
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        
        
        #### mu - ACTOR UPDATE ####
        
        # a_t = mu(s_t)
        action = self.actor(state_batch)
        
        # J = esperance[Q(s_t,mu(s_t))] -> a maximiser
        # -J = policy_loss -> a minimiser
        policy_loss = - self.critic(state_batch, action)
        policy_loss = policy_loss.mean()
        
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        
        
        
        #### update target network with polyak averaging
        soft_update(self.critic_target, self.critic, tau = 0.001)
        soft_update(self.actor_target, self.actor, tau = 0.001)
        return
        

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)






env = gym.make('Pendulum-v0')

nb_states = env.observation_space.shape[0]
nb_actions = env.action_space.shape[0]
agent = Agent(nb_states, nb_actions)

memory = ReplayMemory(100000)
batch_size = 64
epochs = 20000
max_episode_length = 500 # not specified when done by env.step 

epsilon = 1
rewards = []

for i in range(epochs):
    
    obs = env.reset()
    done = False
    total_reward = 0
    episode_count = 0
    epsilon *= 0.99
    while not (done and episode_count < max_episode_length) :
        
        epsilon = max(epsilon, 0.1)
        obs_input = Variable(torch.from_numpy(obs).type(torch.FloatTensor)).type(FLOAT)
        action = agent.act(obs_input, epsilon)
        next_obs, reward, done, _ = env.step(action)
        memory.push(obs_input.data.view(1,-1), torch.Tensor(action), 
                    torch.from_numpy(next_obs).type(torch.FloatTensor).view(1,-1), torch.Tensor([reward]),
                   torch.Tensor([done]))
        obs = next_obs
        total_reward += reward
        episode_count += 1
    rewards.append(total_reward)
    if memory.__len__() > 10000:
        if (i%100 == 0):
            print(i)
            print(total_reward)
        batch = memory.sample(batch_size) 
        agent.backward(batch)
        
pd.DataFrame(rewards).rolling(50, center=False).mean().plot()
#from matplotlib2tikz import save as tikz_save
#tikz_save("rewards.tex")