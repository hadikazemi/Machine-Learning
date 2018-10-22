import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import numpy as np
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'new_state', 'reward', 'done'))

class DQN(nn.Module):
    def __init__(self, layers_size, device):
        super(DQN, self).__init__()
        layers = []
        for l in range(len(layers_size)-1):
            layers += [nn.Linear(layers_size[l], layers_size[l+1])]
            if l < len(layers_size)-2:
                layers += [nn.LeakyReLU()]
        self.model = nn.Sequential(*layers)
        self.device = device

    def forward(self, input):
        return self.model(input.to(self.device))


class Replay:
    def __init__(self, memory_size, state_size=4, device='cpu'):
        self.size = memory_size
        self.memory = []
        self.position = 0

        self.state_size = state_size
        self.device = device

    def push(self, sample):
        if len(self.memory) < self.size:
            self.memory.append(sample)
        else:
            self.memory[self.position] = sample
        self.position += 1
        self.position %= self.size

    def sample(self, batch_size):
        samples = random.sample(self.memory, batch_size)
        samples = np.array(samples)

        state = Variable(torch.FloatTensor(samples[:, :self.state_size])).to(self.device)
        action = Variable(torch.LongTensor(samples[:, self.state_size:self.state_size+1])).to(self.device)
        new_state = Variable(torch.FloatTensor(samples[:, self.state_size+1:2*self.state_size+1])).to(self.device)
        reward = Variable(torch.FloatTensor(samples[:, 2*self.state_size+1:2*self.state_size+2])).to(self.device)
        done = Variable(torch.FloatTensor(samples[:, 2*self.state_size+2:2*self.state_size+3])).to(self.device)

        batch = Transition(state, action, new_state, reward, done)

        return batch


class Policy:
    def __init__(self, dqn, eps_init, eps_min, decay, action_size):
        self.eps = eps_init
        self.eps_min = eps_min
        self.decay = decay
        self.dqn = dqn
        self.action_size = action_size

    def action(self, state, test=False):
        if random.random() > self.eps or test:
            state = Variable(torch.FloatTensor(state).unsqueeze(0))
            action = torch.argmax(self.dqn(state), 1).squeeze().to('cpu').data.numpy()
            action = np.asscalar(action)
        else:
            action = random.randint(0, self.action_size-1)
        return action

    def update_eps(self):
        self.eps = max([(self.eps * self.decay), self.eps_min])
