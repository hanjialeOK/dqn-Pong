import torch

# device
Device = torch.device('cpu')

# environment
Env_name1 = 'LunarLander-v2'
Env_name2 = 'Pong-v0'

# agent parameters
Batch_size = 32
Learning_rate = 0.001
Tau = 0.9
Gamma = 0.99

# training parameters
Num_episode = 3000
Eps_init = 1
Eps_decay = 0.995
Eps_min = 0.05
Max_t = 1500
Num_frame = 2
