import random
from collections import deque
import torch
import torch.optim as optim
import numpy as np
import gym
from tqdm import tqdm

from networks import Q_Network
from replay_memory import ReplayMemory
from history import History

def preprocess(image):
    image = image[34:194, :, :]
    image = image[::2, ::2]
    image = np.mean(image, axis=2, keepdims=False)
    image = image/256
    return image

class Agent:
    def __init__(self, config):
        self.max_step = config.max_step
        self.learn_start = config.learn_start
        self.batch_size = config.batch_size
        self.history_length = config.history_length
        self.learning_rate = config.learning_rate
        self.discount = config.discount
        self.min_reward = config.min_reward
        self.max_reward = config.max_reward
        self.train_frequency = config.train_frequency
        self.target_q_update_step = config.target_q_update_step
        self.ep_end = config.ep_end
        self.ep_start = config.ep_start
        self.ep_end_t = config.ep_end_t
        self.device = torch.device("cuda:0")
        self.Q_local = Q_Network(self.history_length, 2).to(self.device)
        self.Q_target = Q_Network(self.history_length, 2).to(self.device)
        self.update_target()
        self.optimizer = optim.Adam(self.Q_local.parameters(), self.learning_rate)
        self.memory = ReplayMemory(config)
        self.history = History(config)
        self.env = gym.make('Pong-v0')

    def e_greedy(self, states, eps):
        # TODO: update ep
        if random.random() > eps:
            states = torch.tensor(states, dtype=torch.float32).to(self.device)
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
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        terminals = torch.tensor(terminals, dtype=torch.float32).to(self.device)

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

    def update_target(self, tau=1.):
        for local_param, target_param in zip(self.Q_local.parameters(), self.Q_target.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    def train(self):
        num_game, update_count, episode_reward = 0, 0, 0
        total_reward, total_loss, total_q = 0., 0., 0.
        episode_rewards = []

        screen = self.env.reset()
        screen = preprocess(screen)
        for _ in range(self.history_length):
            self.history.add(screen)

        for step in tqdm(range(0, self.max_step), ncols=70, initial=0):
            if step == self.learn_start:
                num_game, update_count, episode_reward = 0, 0, 0
                total_reward, total_loss, total_q = 0., 0., 0.
                episode_rewards = []
            
            # TODO: update ep. 1. predict
            eps = self.ep_end + max(0., (self.ep_start - self.ep_end)*(self.ep_end_t - max(0., step - self.learn_start)) / self.ep_end_t)
            action = self.e_greedy(self.history.get(), eps)
            # 2. act
            screen, reward, terminal, _ = self.env.step(action)
            screen = preprocess(screen)
            # 3. observe

            reward = max(self.min_reward, min(self.max_reward, reward))
            self.history.add(screen)
            self.memory.add(action, screen, reward, terminal)
            episode_reward += reward

            if step >= self.learn_start:
                if step % self.train_frequency == 0:
                    self.update_local()
                if step % self.target_q_update_step == self.target_q_update_step - 1:
                    self.update_target()

            if terminal == True:
                print("episode: %d, reward: %f, is learning: %d" % (num_game, episode_reward, step >= self.learn_start))
                episode_rewards.append(episode_reward)
                episode_reward = 0;
                num_game += 1
                screen = self.env.reset()
                screen = preprocess(screen)
                for _ in range(self.history_length):
                    self.history.add(screen)