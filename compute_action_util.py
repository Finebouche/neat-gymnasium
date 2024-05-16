import numpy as np


def compute_action_lander(net, observation):
    activation = net.activate(observation)
    # Gym expects discrete actions (0, 1, 2, 3), so we need to convert the output
    return np.argmax(activation)


def compute_action_bipedal_walker(net, observation):
    activation = net.activate(observation)
    return activation
