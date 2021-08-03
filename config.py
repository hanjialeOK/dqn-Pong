class Config:
    scale = 10000
    display = False

    max_step = 5000 * scale
    memory_size = 100 * scale

    batch_size = 32
    random_start = 30
    cnn_format = 'NCHW'
    discount = 0.99
    target_q_update_step = 1 * scale
    learning_rate = 0.00025
    learning_rate_minimum = 0.00025
    learning_rate_decay = 0.96
    learning_rate_decay_step = 5 * scale

    ep_end = 0.1
    ep_start = 1.
    ep_end_t = memory_size

    history_length = 4
    train_frequency = 4
    learn_start = 5. * scale

    min_delta = -1
    max_delta = 1

    double_q = False
    dueling = False

    _test_step = 5 * scale
    _save_step = _test_step * 10

    screen_width  = 80
    screen_height = 80
    max_reward = 1.
    min_reward = -1.