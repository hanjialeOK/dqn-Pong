import random
from collections import deque
import torch
import torch.optim as optim
import numpy as np
import gym
import tqdm

from networks import Q_Network
from replay_memory import ReplayMemory
from history import History

class Agent:
    def __init__(self, config):
        self.max_step = config.max_step
        self.learn_start = config.learn_start
        self.batch_size = config.batch_size
        self.history_length = config.history_length
        self.learning_rate = config.learning_rate
        self.discount = config.discount
        self.min_reward = config.min_rewrad
        self.max_reward = config.max_reward
        self.train_frequency = config.train_frequency
        self.target_q_update_step = config.target_q_update_step
        self.device = device
        self.Q_local = Q_Network(self.state_size, self.action_size).to(self.device)
        self.Q_target = Q_Network(self.state_size, self.action_size).to(self.device)
        self.soft_update(1)
        self.optimizer = optim.Adam(self.Q_local.parameters(), self.learning_rate)
        self.memory = ReplayMemory(config)
        self.history = History(config)
        self.env = gym.make('Pong-v0')

    def e_greedy(self, states, eps):
        # TODO: update ep
        if random.random() > eps:
            states = torch.tensor(states).to(self.device)
            with torch.no_grad():
                action_values = self.Q_local(states)
            return np.argmax(action_values.cpu().data.numpy()) + 2
        else:
            return random.choice(np.arange(2, 4))

    def update_local(self):
        # experiences = random.sample(self.memory, self.bs)
        # states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float().to(self.device)
        # actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).long().to(self.device)
        # rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float().to(self.device)
        # next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float().to(self.device)
        # dones = torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8)).float().to(self.device)

        states, actions, rewards, next_states, terminals = self.memory.sample()
        # np.float16 -> torch.float16
        states = torch.tensor(states).to(self.device)
        # np.uint8 -> torch.uint8
        actions = torch.tensor(actions).to(self.device)
        # np.int8 -> torch.int8
        rewards = torch.tensor(rewards).to(self.device)
        # np.float16 -> torch.float16
        next_states = torch.tensor(next_states).to(self.device)
        # np.bool_ -> torch.bool
        terminals = torch.tensor(terminals).to(self.device)

        q_values = self.Q_local(states)
        # print(actions.view(-1, 32))
        q_values = torch.gather(input=q_values, dim=-1, index=actions-2)

        with torch.no_grad():
            q_targets = self.Q_target(next_states)
            q_targets, _ = torch.max(input=q_targets, dim=-1, keepdim=True)
            q_targets = rewards + self.discount * (1 - terminals) * q_targets

        loss = (q_targets - q_values).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self, tau):
        for local_param, target_param in zip(self.Q_local.parameters(), self.Q_target.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    def train(self):
        screen = self.env.reset()
        for _ in range(self.history_length):
            self.history.add(screen)

        for step in tqdm(range(0, self.max_step), ncols=70):
            if step == self.learn_start:
                num_game, update_count, ep_reward = 0, 0, 0
                total_reward, total_loss, total_q = 0., 0., 0.
            
            # TODO: update ep. 1. predict
            action = self.e_greedy(self.history)
            # 2. act
            screen, reward, terminal, _ = self.env.step(action)
            # 3. observe

            reward = max(self.min_reward, min(self.max_reward, reward))
            self.history.add(screen)
            self.memory.add(screen, reward, action, terminal)

            if step >= self.learn_start:
                if step % self.train_frequency == 0:
                    self.update_local()
                if step % self.target_q_update_step == self.target_q_update_step - 1:
                    self.update_target()

            