# Evolve a control/reward estimation network for the OpenAI Gym
# LunarLander-v2 environment (https://gym.openai.com/envs/LunarLander-v2).
# Sample run here: https://gym.openai.com/evaluations/eval_FbKq5MxAS9GlvB7W6ioJkg


import multiprocessing
import os
import pickle
import time

import gymnasium

import neat
import visualize
from compute_action_util import compute_action_discrete, compute_action_box

NUM_CORES = multiprocessing.cpu_count()


class PooledErrorCompute(object):
    def __init__(self, num_workers, env_name, compute_action):
        self.num_workers = num_workers
        self.generation = 0
        self.env = gymnasium.make(env_name)
        self.compute_action = compute_action

    def compute_reward(self, net):
        try:
            total_reward = 0.0
            for n in range(5):
                observation_init_vals, observation_init_info = self.env.reset()

                t = 0
                while t < 1000:
                    action = self.compute_action(net, observation_init_vals)
                    observation, reward, terminated, done, info = self.env.step(action)

                    total_reward += reward
                    t += 1
                    if terminated or done:
                        break

            return total_reward / 5

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


def run(config_file="config-default", env_name="LunarLander-v2", num_generations=None, checkpoint=None):
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config", config_file)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Load the population if checkpoint is not None
    if checkpoint is not None:
        pop = neat.Checkpointer.restore_checkpoint(checkpoint)
    else:
        pop = neat.Population(config)

    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.Checkpointer(generation_interval=100000, time_interval_seconds=7000))

    # Create env en check if it is discrete or box
    env = gymnasium.make(env_name)
    if isinstance(env.action_space, gymnasium.spaces.Discrete):
        compute_action_function = compute_action_discrete
    else:
        compute_action_function = compute_action_box
    env.close()

    ec = PooledErrorCompute(NUM_CORES, env_name, compute_action_function)

    # Run until the winner from a generation is able to solve the environment
    gen_best = pop.run(ec.eval_genomes, num_generations)

    # Display the winning genome.
    visualization_path = os.path.join(local_dir, "visualisations", env_name)
    visualize.draw_net(config, gen_best, True, filename=visualization_path + "/win-net.gv")
    visualize.plot_stats(stats, ylog=False, view=True, filename=visualization_path + "/avg_fitness.svg")

    # Save the best mosel
    name = 'winner-net'
    with open(name + '.pickle', 'wb') as f:
        pickle.dump(gen_best, f)


if __name__ == '__main__':
    # https://github.com/ShangtongZhang/DistributedES/blob/master/neat-config/BipedalWalker-v2.txt
    # LunarLander-v2 CarRacing-v1, BipedalWalker-v3, CartPole-v1
    run(config_file="config-lander", env_name='LunarLander-v2', num_generations=10e0)
