import gym
import numpy as np
from collections import deque
from agent import Agent
from config import *

def preprocess(image):
    image = image[34:194, :, :]
    image = image[::2, ::2]
    image = np.mean(image, axis=2, keepdims=False)
    image = image/256
    return image

def train(env, agent, num_episode, eps_init, eps_decay, eps_min, max_t, num_frame=4):
    rewards_log = []
    average_log = []
    eps = eps_init

    for i in range(1, 1 + num_episode):
        episodic_reward = 0
        done = False
        frame = env.reset()
        frame = preprocess(frame)
        state_deque = deque(maxlen=num_frame)
        for _ in range(num_frame):
            state_deque.append(frame)
        state = np.stack(state_deque, axis=0)
        state = np.expand_dims(state, axis=0)
        t = 0

        while not done and t < max_t:
            t += 1
            action = agent.e_greedy(state, eps)
            frame, reward, done, _ = env.step(action)
            frame = preprocess(frame)
            state_deque.append(frame)
            next_state = np.stack(state_deque, axis=0)
            next_state = np.expand_dims(next_state, axis=0)
            agent.memory.append((state, action, reward, next_state, done))

            if t % 5 == 0 and len(agent.memory) >= agent.bs:
                agent.learn()
                agent.soft_update(agent.tau)

            state = next_state.copy()
            episodic_reward += reward
        
        rewards_log.append(reward)
        average_log.append(np.mean(rewards_log[-100:]))
        print('\rEpisode {}, Reward {:.3f}, Average Reward {:.3f}'.format(i, episodic_reward, average_log[-1]), end='')

        if t % 100 == 0:
            print()

        eps = max(eps * eps_decay, eps_min)
    
    return rewards_log

if __name__ == '__main__':
    env = gym.make(Env_name2)
    agent = Agent(Num_frame, 2, Batch_size, Learning_rate, Tau, Gamma, Device)
    rewards_log = train(env, agent, Num_episode, Eps_init, Eps_decay, Eps_min, Max_t, Num_frame)
    agent.Q_local.to('cpu')
    torch.save(agent.Q_local.state_dict(), '{}.pth'.format(Env_name2))