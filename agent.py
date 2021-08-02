import random
from collections import deque
import torch
import torch.optim as optim
import numpy as np

from networks import Q_Network
from replay_memory import ReplayMemory
from history import History

class Agent:
    def __init__(self, config):
        self.batch_size = config.batch
        self.lr = lr
        self.tau = tau
        self.gamma = gamma
        self.device = device
        self.Q_local = Q_Network(self.state_size, self.action_size).to(self.device)
        self.Q_target = Q_Network(self.state_size, self.action_size).to(self.device)
        self.soft_update(1)
        self.optimizer = optim.Adam(self.Q_local.parameters(), self.lr)
        self.memory = deque(maxlen=200000)

    def e_greedy(self, state, eps):
        if random.random() > eps:
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                action_values = self.Q_local(state)
            return np.argmax(action_values.cpu().data.numpy())+2
        else:
            return random.choice(np.arange(2, 4))

    def learn(self):
        experiences = random.sample(self.memory, self.bs)
        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8)).float().to(self.device)

        q_values = self.Q_local(states)
        # print(actions.view(-1, 32))
        q_values = torch.gather(input=q_values, dim=-1, index=actions-2)

        with torch.no_grad():
            q_targets = self.Q_target(next_states)
            q_targets, _ = torch.max(input=q_targets, dim=-1, keepdim=True)
            q_targets = rewards + self.gamma * (1 - dones) * q_targets

        loss = (q_targets - q_values).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def soft_update(self, tau):
        for local_param, target_param in zip(self.Q_local.parameters(), self.Q_target.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)
