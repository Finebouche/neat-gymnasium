import numpy as np
import neat
import gymnasium


def compute_action_discrete(net, observation):
    activation = net.activate(observation)
    # Gym expects discrete actions (0, 1, 2, 3), so we need to convert the output
    action = np.argmax(activation)
    return action, activation[action]


def compute_action_box(net, observation):
    action = net.activate(observation)
    # compute the norm of the action array
    norm = np.linalg.norm(action)
    return action, norm


def compute_reward(genome, config, env_name, env_args, penalize_inactivity, num_episodes=3):
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
            action, norm = compute_action(net, observation)

            # to avoid local minima were the agent does not move
            if norm < 0.5 and penalize_inactivity:
                total_reward -= 1
            observation, reward, terminated, done, info = env.step(action)

            total_reward += reward
            if terminated or done:
                break

    env.close()
    return total_reward / num_episodes
