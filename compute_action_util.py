import numpy as np
import neat
import gymnasium


def compute_action_discrete(net, observation):
    activation = net.activate(observation)
    # Gym expects discrete actions (0, 1, 2, 3), so we need to convert the output
    action = np.argmax(activation)
    return action


def compute_action_box(net, observation):
    action = net.activate(observation)
    return action


def compute_reward(genome, config, env_name, env_args, num_episodes=3):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    env = gymnasium.make(env_name, **env_args)
    if isinstance(env.action_space, gymnasium.spaces.Discrete):
        compute_action = compute_action_discrete
    else:
        compute_action = compute_action_box

    total_reward = 0.0
    for n in range(num_episodes):
        observation, observation_init_info = env.reset()

        while True:
            action = compute_action(net, observation)
            observation, reward, terminated, done, info = env.step(action)

            total_reward += reward
            if terminated or done:
                break

    env.close()
    return total_reward / num_episodes
