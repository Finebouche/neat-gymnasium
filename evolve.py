# Evolve a control/reward estimation network for the OpenAI Gym
# LunarLander-v2 environment (https://gym.openai.com/envs/LunarLander-v2).
# Sample run here: https://gym.openai.com/evaluations/eval_FbKq5MxAS9GlvB7W6ioJkg


import multiprocessing
import os
import pickle
import time

import gymnasium.wrappers
import numpy as np

import neat
import visualize
from neat.nn import FeedForwardNetwork

NUM_CORES = multiprocessing.cpu_count()


class VectorizedFeedForwardNetwork(FeedForwardNetwork):
    def __init__(self, inputs, outputs, node_evals):
        super(VectorizedFeedForwardNetwork, self).__init__(inputs, outputs, node_evals)

    def activate(self, inputs):
        if inputs.shape[1] != len(self.input_nodes):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), inputs.shape[1]))

        num_samples = inputs.shape[0]
        self.values[:len(self.input_nodes)] = inputs.T

        for node, act_func, agg_func, bias, response, links in self.node_evals:
            node_inputs = np.zeros((num_samples, len(links)))
            for j, (i, w) in enumerate(links):
                node_inputs[:, j] = self.values[i] * w
            s = agg_func(node_inputs, axis=1)
            self.values[node] = act_func(bias + response * s)

        return self.values[self.output_nodes]


def compute_action_lander(net, observation):
    activation = net.activate(observation)
    # Gym expects discrete actions (0, 1, 2, 3), so we need to convert the output
    return np.argmax(activation)


def compute_action_bipedal_walker(net, observation):
    activation = net.activate(observation)
    return activation


class PooledErrorCompute(object):
    def __init__(self, num_workers, env_name, compute_action):
        self.num_workers = num_workers
        self.generation = 0
        self.env = gymnasium.make(env_name)
        self.compute_action = compute_action

    def compute_reward(self, net):
        try:
            observation_init_vals, observation_init_info = self.env.reset()
            terminated = False
            t = 0

            total_reward = 0.0
            for n in range(10):
                while not terminated and t < 1000:
                    action = self.compute_action(net, observation_init_vals)
                    observation, reward, terminated, done, info = self.env.step(action)

                    total_reward += reward
                    t += 1
                    if terminated or done:
                        break

            return total_reward / 10
        except Exception as e:
            print(f"Error in compute_reward: {e}")
            return 0.0

    def eval_genomes(self, genomes, config):
        t0 = time.time()
        self.generation += 1
        if self.num_workers < 2:
            for _, genome in genomes:
                net = neat.nn.FeedForwardNetwork.create(genome, config)

                reward_error = self.compute_reward(net)
                genome.fitness = reward_error
        else:
            with multiprocessing.Pool(self.num_workers) as pool:
                jobs = []
                for _, genome in genomes:
                    net = neat.nn.FeedForwardNetwork.create(genome, config)

                    jobs.append(pool.apply_async(
                        self.compute_reward,
                        [net]
                    ))

                for job, (_, genome) in zip(jobs, genomes):
                    try:
                        reward = job.get(timeout=None)
                        genome.fitness = reward
                    except multiprocessing.TimeoutError:
                        print(f"TimeoutError in generation {self.generation} for genome {genome.key}")
                        genome.fitness = 0.0
                    except Exception as e:
                        print(f"Error in generation {self.generation} for genome {genome.key}: {e}")
                        genome.fitness = 0.0
        print("final fitness compute time {0}\n".format(time.time() - t0))

    # use function to close env when done
    def __delete__(self):
        self.env.close()


def run(config_file="config", env_name="LunarLander-v2", num_generations=None, checkpoint=None):
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, config_file)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    if checkpoint is not None:
        pop = neat.Checkpointer.restore_checkpoint(checkpoint)
    else:
        pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    # Checkpoint every 25 generations or 900 seconds.
    pop.add_reporter(neat.Checkpointer(1000))  # neat.Checkpointer(25, 900)

    # Run until the winner from a generation is able to solve the environment
    # or the user interrupts the process.
    ec = PooledErrorCompute(NUM_CORES, env_name, compute_action_bipedal_walker)

    gen_best = pop.run(ec.eval_genomes, num_generations)
    visualize.draw_net(config, gen_best, True, filename="win-net.gv")
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    name = 'winner-net'
    with open(name + '.pickle', 'wb') as f:
        pickle.dump(gen_best, f)



if __name__ == '__main__':
    # https://github.com/ShangtongZhang/DistributedES/blob/master/neat-config/BipedalWalker-v2.txt
    # LunarLander-v2 CarRacing-v1, BipedalWalker-v3
    run(config_file="config-walker", env_name="BipedalWalker-v3", num_generations=2000)
