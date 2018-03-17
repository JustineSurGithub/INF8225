# based on http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import gym
from torch.autograd import Variable
import random
from collections import namedtuple


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q


# from https://github.com/ghliu/pytorch-ddpg/blob/master/util.py

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class Agent(object):
    def __init__(self, gamma=0.99, batch_size_=128):
        self.target_Q = DQN()
        self.Q = DQN()
        self.gamma = gamma
        self.duelling = True
        self.batch_size_ = batch_size_
        self.lr = 0.001
        hard_update(self.target_Q, self.Q)
        self.optimizer = torch.optim.Adam(self.Q.parameters(), self.lr)

    def act(self, x, epsilon_=0.1):
        if random.random() < epsilon_:
            return Variable(torch.from_numpy(np.array([env.action_space.sample()])).type(torch.LongTensor))
        else:
            value, indice = torch.max(self.Q.forward(x), 0)
            return indice

    def backward(self, transitions):
        my_batch = Transition(*zip(*transitions))
        state_batch = Variable(torch.cat(my_batch.state))
        next_state_batch = Variable(torch.cat(my_batch.next_state))
        action_batch = Variable(torch.cat(my_batch.action))
        reward_batch = Variable(torch.cat(my_batch.reward))
        done_batch = Variable(torch.cat(my_batch.done))
        state_action_values = self.Q(state_batch).gather(1, action_batch.view(self.batch_size_, 1))

        mask = np.logical_not(done_batch.data.numpy()) * np.ones(self.batch_size_)
        mask = Variable(torch.from_numpy(mask).type(torch.FloatTensor))
        if self.duelling is False:
            next_state_action_values = [self.target_Q.forward(Variable(j)).max().data for j in my_batch.next_state]
            next_state_action_values = Variable(torch.cat(next_state_action_values))
        else:
            next_state_actions = [self.Q.forward(Variable(j)).data for j in my_batch.next_state]
            values, next_state_actions = Variable(torch.cat(next_state_actions)).detach().max(1)
            next_state_action_values = self.target_Q(next_state_batch).gather(
                1, next_state_actions.view(self.batch_size_, 1)).view(-1).detach()

        expected_state_action_values = next_state_action_values * self.gamma * mask + reward_batch
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        soft_update(self.target_Q, self.Q, self.lr)


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

    def sample(self, batch_size_):
        return random.sample(self.memory, batch_size_)

    def __len__(self):
        return len(self.memory)

env = gym.make('CartPole-v0')
agent = Agent()
memory = ReplayMemory(100000)
batch_size = 128

epsilon = 1
rewards = []

for i in range(5000):
    obs = env.reset()
    done = False
    total_reward = 0
    epsilon *= 0.99
    while not done:
        epsilon = max(epsilon, 0.1)
        obs_input = Variable(torch.from_numpy(obs).type(torch.FloatTensor))
        action = agent.act(obs_input, epsilon)
        next_obs, reward, done, _ = env.step(action.data.numpy()[0])
        memory.push(obs_input.data.view(1, -1), action.data,
                    torch.from_numpy(next_obs).type(torch.FloatTensor).view(1, -1), torch.Tensor([reward]),
                    torch.Tensor([done]))
        obs = next_obs
        total_reward += reward
    rewards.append(total_reward)
    if memory.__len__() > 10000:
        batch = memory.sample(batch_size)
        agent.backward(batch)

pd.DataFrame(rewards).rolling(50, center=False).mean().plot().get_figure().savefig("output.png")
