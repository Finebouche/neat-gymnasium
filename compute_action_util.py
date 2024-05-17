import numpy as np


def compute_action_discrete(net, observation):
    activation = net.activate(observation)
    # Gym expects discrete actions (0, 1, 2, 3), so we need to convert the output
    action = np.argmax(activation)
    return action


def compute_action_box(net, observation):
    action = net.activate(observation)
    return action
