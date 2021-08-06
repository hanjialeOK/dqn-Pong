import random
from collections import deque
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import time
import os
import matplotlib.pyplot as plt

from networks import Q_Network
from replay_memory import ReplayMemory
from history import History
from environment import Environment

class Agent:
    def __init__(self, config):
        self.num_action = config.num_action
        self.max_step = config.max_step
        self.test_step = config._test_step
        self.save_step = config._save_step
        self.learn_start = config.learn_start
        self.batch_size = config.batch_size
        self.history_length = config.history_length
        self.screen_height = config.screen_height
        self.screen_width = config.screen_width
        self.learning_rate = config.learning_rate
        self.discount = config.discount
        self.train_frequency = config.train_frequency
        self.target_q_update_step = config.target_q_update_step
        self.ep_end = config.ep_end
        self.ep_start = config.ep_start
        self.ep_end_t = config.ep_end_t
        self.eps = self.ep_start
        self.env_name = config.env_name
        self.device = torch.device("cuda:0")
        self.local_network = Q_Network(self.history_length, self.num_action).to(self.device)
        self.target_network = Q_Network(self.history_length, self.num_action).to(self.device)
        self.update_target()
        self.optimizer = optim.Adam(self.local_network.parameters(), self.learning_rate)
        self.memory = ReplayMemory(config)
        self.history = History(config)
        self.env = Environment(config)

    def predict_ep_greedy(self, states):
        # TODO: record eps
        self.eps = self.ep_end + max(0., (self.ep_start - self.ep_end)* \
            (self.ep_end_t - max(0., self.step - self.learn_start)) / self.ep_end_t)
        # ep-greedy
        if random.random() > self.eps:
            # [1, 4, 80, 80]
            states = torch.tensor(states, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                action_values = self.local_network(states)
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.num_action))

    def predict(self, states):
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            action_values = self.local_network(states)
        return np.argmax(action_values.cpu().data.numpy())

    def update_local(self):
        # experiences = random.sample(self.memory, self.bs)
        # states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float().to(self.device)
        # actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).long().to(self.device)
        # rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float().to(self.device)
        # next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float().to(self.device)
        # dones = torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8)).float().to(self.device)

        # random sample
        states, actions, rewards, next_states, terminals = self.memory.sample()
        # [32, 4, 80, 80]
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        # [32, 1]
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        # [32, 1]
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        # [32, 4, 80, 80]
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        # [32, 1]
        assert terminals.shape == (self.batch_size, 1), "terminals.shape is not correct."
        terminals = torch.tensor(terminals, dtype=torch.float32).to(self.device)

        q_values = self.local_network(states)
        q_values = torch.gather(input=q_values, dim=-1, index=actions)
        self.total_q += torch.mean(q_values.detach()).cpu().item()

        with torch.no_grad():
            q_targets = self.target_network(next_states)
            q_targets, _ = torch.max(input=q_targets, dim=-1, keepdim=True)
            q_targets = rewards + self.discount * (1 - terminals) * q_targets

        # loss = (q_targets - q_values).pow(2).mean()
        loss = torch.mean(torch.pow((q_targets - q_values), 2))
        self.total_loss += loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self, tau=1.):
        for local_param, target_param in zip(self.local_network.parameters(), self.target_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    def learn(self):
        if self.step >= self.learn_start:
            if self.step % self.train_frequency == 0:
                self.update_local()
                self.update_count += 1
            if self.step % self.target_q_update_step == self.target_q_update_step - 1:
                self.update_target()

    def save_model(self, dir_name="models"):
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        torch.save(self.local_network.state_dict(), \
            os.path.join(dir_name, "%s_v%d.pth" % (self.env_name, (self.step+1)/self.test_step)))

    def save_checkpoint(self, dir_name="checkpoints"):
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        torch.save(self.local_network.state_dict(), \
            os.path.join(dir_name, "%s_v%d.pth" % (self.env_name, (self.step+1)/self.save_step)))

    def save_record(self, dir_name="records"):
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        doc = open(os.path.join(dir_name, "%s_record_v%d.txt" % (self.env_name, (self.step+1)/self.save_step)), 'w')
        length = len(self.avg_qs)
        for i in range(length):
            print("%f %f %f %f %f" % (self.avg_qs[i], self.max_episode_rewards[i], \
                self.min_episode_rewards[i], self.avg_episode_rewards[i], self.eps), file=doc)
        doc.close()

    def save_figure(self, dir_name="figures"):
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        fig = plt.figure()
        plt.xlabel('step')
        plt.ylabel('avg_q')
        plt.plot(self.avg_qs)
        plt.savefig(os.path.join(dir_name, "%s_avg_q_v%d.pdf" % (self.env_name, (self.step+1)/self.save_step)))

        fig = plt.figure()
        plt.xlabel('episode')
        plt.ylabel('reward')
        plt.plot(self.max_episode_rewards, 'r--', label='max_reward')
        plt.plot(self.min_episode_rewards, 'g--', label='min_reward')
        plt.plot(self.avg_episode_rewards, color='b', label='avg_reward')
        plt.legend()
        plt.savefig(os.path.join(dir_name, "%s_reward_v%d.pdf" % (self.env_name, (self.step+1)/self.save_step)))

    def train(self):
        num_game, self.update_count, episode_reward = 0, 0, 0
        total_reward, self.total_loss, self.total_q = 0., 0., 0.
        max_avg_episode_reward = 0
        episode_rewards = []
        self.avg_qs, self.max_episode_rewards, self.min_episode_rewards, self.avg_episode_rewards = [], [], [], []

        screen = self.env.new_random_game()
        for _ in range(self.history_length):
            self.history.add(screen)

        for self.step in tqdm(range(0, self.max_step), ncols=70, initial=0):
            # initialize
            if self.step == self.learn_start:
                num_game, self.update_count, episode_reward = 0, 0, 0
                total_reward, self.total_loss, self.total_q = 0., 0., 0.
                episode_rewards = []
            
            # 1. predict
            action = self.predict_ep_greedy(self.history.get())

            # 2. act
            screen, reward, terminal, _ = self.env.step(action)

            # 3. memorize
            self.memory.add(action, screen, reward, terminal)

            # 4. learn
            self.learn()

            episode_reward += reward
            total_reward += reward

            if terminal == True:
                screen = self.env.new_random_game()
                for _ in range(self.history_length):
                    self.history.add(screen)

                print("episode: %d, mem_size: %d, eps: %f, reward: %d" % \
                    (num_game, self.memory.count, self.eps, episode_reward))
                episode_rewards.append(episode_reward)
                episode_reward = 0
                num_game += 1
            else:
                self.history.add(screen)

            if self.step >= self.learn_start:
                if self.step % self.test_step == self.test_step - 1:
                    avg_reward = total_reward / self.test_step
                    avg_loss = self.total_loss / self.update_count
                    avg_q = self.total_q / self.update_count

                    try:
                        max_episode_reward = np.max(episode_rewards)
                        min_episode_reward = np.min(episode_rewards)
                        avg_episode_reward = np.mean(episode_rewards)
                    except:
                        max_episode_reward, min_episode_reward, avg_episode_reward = 0., 0., 0.

                    self.avg_qs.append(avg_q)
                    self.max_episode_rewards.append(max_episode_reward)
                    self.min_episode_rewards.append(min_episode_reward)
                    self.avg_episode_rewards.append(avg_episode_reward)

                    if avg_episode_reward >= max_avg_episode_reward * 0.9:
                        self.save_model("models")
                    max_avg_episode_reward = max(max_avg_episode_reward, avg_episode_reward)

                    total_reward = 0
                    self.total_loss = 0
                    self.total_q = 0
                    self.update_count =  0
                    episode_rewards = []

                if self.step % self.save_step == self.save_step - 1:
                    self.save_checkpoint("checkpoints")
                    self.save_record("records")
                    self.save_figure("figures")

    def play(self):

        print("loading weight...")
        self.local_network.to('cpu')
        self.local_network.load_state_dict(torch.load("%s_v43.pth" % (self.env_name)))
        self.local_network.to(self.device)
        print("weight loaded successfully.")

        screen = self.env.new_random_game()
        for _ in range(self.history_length):
            self.history.add(screen)

        episode_reward, num_game = 0, 0

        for step in tqdm(range(0, 100000), ncols=70, desc="%s playing" % (self.env_name)):
            # 1. predict
            action = self.predict(self.history.get())

            # 2. act
            screen, reward, terminal, _ = self.env.step(action)

            # 3. render
            self.env.render()

            episode_reward += reward

            if terminal == True:
                self.env.new_random_game()
                for _ in range(self.history_length):
                    self.history.add(screen)

                print("episode: %d, reward: %f" % (num_game, episode_reward))
                episode_reward = 0
                num_game += 1
            else:
                self.history.add(screen)

            # time.sleep(0.01)
        
        self.env.close()