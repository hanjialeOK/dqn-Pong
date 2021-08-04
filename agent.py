import random
from collections import deque
from numpy.core.fromnumeric import argmax
import torch
import torch.optim as optim
import numpy as np
import gym
from tqdm import tqdm
import time

from networks import Q_Network
from replay_memory import ReplayMemory
from history import History
from environment import Environment

class Agent:
    def __init__(self, config):
        self.max_step = config.max_step
        self.test_step = config._test_step
        self.save_step = config._save_step
        self.learn_start = config.learn_start
        self.batch_size = config.batch_size
        self.history_length = config.history_length
        self.learning_rate = config.learning_rate
        self.discount = config.discount
        self.train_frequency = config.train_frequency
        self.target_q_update_step = config.target_q_update_step
        self.ep_end = config.ep_end
        self.ep_start = config.ep_start
        self.ep_end_t = config.ep_end_t
        self.device = torch.device("cuda:1")
        self.Q_local = Q_Network(self.history_length, 2).to(self.device)
        self.Q_target = Q_Network(self.history_length, 2).to(self.device)
        self.update_target()
        self.optimizer = optim.Adam(self.Q_local.parameters(), self.learning_rate)
        self.memory = ReplayMemory(config)
        self.history = History(config)
        self.env = Environment(config)

    def e_greedy(self, states):
        # update ep
        eps = self.ep_end + max(0., (self.ep_start - self.ep_end)* \
            (self.ep_end_t - max(0., self.step - self.learn_start)) / self.ep_end_t)
        # ep-greedy
        if random.random() > eps:
            states = torch.tensor(states, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                action_values = self.Q_local(states)
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(2))

    def update_local(self):
        # experiences = random.sample(self.memory, self.bs)
        # states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float().to(self.device)
        # actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).long().to(self.device)
        # rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float().to(self.device)
        # next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float().to(self.device)
        # dones = torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8)).float().to(self.device)

        states, actions, rewards, next_states, terminals = self.memory.sample()
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        # [batch_size] -> [batch_size, 1]
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(-1).to(self.device)
        # [batch_size] -> [batch_size, 1]
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        terminals = torch.tensor(terminals, dtype=torch.float32).to(self.device)

        q_values = self.Q_local(states)
        q_values = torch.gather(input=q_values, dim=-1, index=actions)
        self.total_q += torch.mean(q_values)

        with torch.no_grad():
            q_targets = self.Q_target(next_states)
            q_targets, _ = torch.max(input=q_targets, dim=-1, keepdim=True)
            q_targets = rewards + self.discount * (1 - terminals) * q_targets

        # loss = (q_targets - q_values).pow(2).mean()
        loss = torch.mean(torch.pow((q_targets - q_values), 2))
        self.total_loss += loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self, tau=1.):
        for local_param, target_param in zip(self.Q_local.parameters(), self.Q_target.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    def train(self):
        num_game, update_count, episode_reward = 0, 0, 0
        total_reward, self.total_loss, self.total_q = 0., 0., 0.
        episode_rewards = []
        doc = open("record.txt", 'w')

        screen = self.env.reset()
        for _ in range(self.history_length):
            self.history.add(screen)

        for self.step in tqdm(range(0, self.max_step), ncols=70, initial=0, desc="Pong-v0 training "):
            # initialize
            if self.step == self.learn_start:
                num_game, update_count, episode_reward = 0, 0, 0
                total_reward, self.total_loss, self.total_q = 0., 0., 0.
                episode_rewards = []
            
            # 1. predict
            action = self.e_greedy(self.history.get())

            # 2. act
            screen, reward, terminal, _ = self.env.step(action)

            # 3. memorize
            self.memory.add(action, screen, reward, terminal)

            # 4. learn
            if self.step >= self.learn_start:
                if self.step % self.train_frequency == 0:
                    self.update_local()
                    update_count += 1
                if self.step % self.target_q_update_step == self.target_q_update_step - 1:
                    self.update_target()

            episode_reward += reward
            total_reward += reward

            if terminal == True:
                print("episode: %d, reward: %f" % (num_game, episode_reward))
                episode_rewards.append(episode_reward)
                episode_reward = 0
                num_game += 1
                screen = self.env.reset()
                for _ in range(self.history_length):
                    self.history.add(screen)
            else:
                self.history.add(screen)

            if self.step >= self.learn_start:
                if self.step % self.test_step == self.test_step - 1:
                    avg_reward = total_reward / self.test_step
                    avg_loss = self.total_loss / update_count
                    avg_q = self.total_q / update_count

                    try:
                        max_episode_reward = np.max(episode_rewards)
                        min_episode_reward = np.min(episode_rewards)
                        avg_episode_reward = np.mean(episode_rewards)
                    except:
                        max_episode_reward, min_episode_reward, avg_episode_reward = 0., 0., 0.

                    # print("avg_episode_reward: %f" % (avg_episode_reward))
                    print("%f %f %f %f %f %f" % (avg_reward, avg_loss, avg_q, max_episode_reward, min_episode_reward, avg_episode_reward), file=doc)

                    total_reward = 0
                    self.total_loss = 0
                    self.total_q = 0
                    update_count =  0
                    episode_rewards = []

                if self.step % self.save_step == 0:
                    # self.Q_local.to('cpu')
                    torch.save(self.Q_local.state_dict(), "{}_v%d.pth".format("Pong-v0") % (self.step/self.save_step))
                    # self.Q_local.to(self.device)

        # self.Q_local.to('cpu')
        torch.save(self.Q_local.state_dict(), "{}_final.pth".format("Pong-v0"))
        doc.close()

    def play(self):

        print("loading weight...")
        self.Q_local.to('cpu')
        self.Q_local.load_state_dict(torch.load("Pong-v0_v11.pth"))
        self.Q_local.to(self.device)
        print("weight loaded successfully.")

        screen = self.env.reset()
        for _ in range(self.history_length):
            self.history.add(screen)

        episode_reward, num_game = 0, 0

        for step in tqdm(range(0, 100000), ncols=70, desc="Pong-v0 playing"):
            # 1. predict
            states = torch.tensor(self.history.get(), dtype=torch.float32).to(self.device)
            with torch.no_grad():
                predict = self.Q_local(states)
            action = np.argmax(predict.cpu().data.numpy())

            # 2. act
            screen, reward, terminal, _ = self.env.step(action)

            # 3. render
            self.env.render()

            episode_reward += reward

            if terminal == True:
                print("episode: %d, reward: %f" % (num_game, episode_reward))
                episode_reward = 0
                num_game += 1
                self.env.reset()
                for _ in range(self.history_length):
                    self.history.add(screen)
            else:
                self.history.add(screen)

            time.sleep(0.01)
        
        self.env.close()