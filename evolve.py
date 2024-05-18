from multiprocessing import Pool, cpu_count, TimeoutError
import os
import pickle
import time
import gymnasium
import neat
import visualize
from compute_action_util import compute_action_discrete, compute_action_box, compute_reward

NUM_CORES = cpu_count()


class ParallelRewardEvaluator(object):
    def __init__(self, num_workers, env_name, env_args):
        self.num_workers = num_workers
        self.generation = 0
        self.pool = Pool(processes=num_workers)
        self.env = gymnasium.make(env_name, **env_args)
        if isinstance(self.env.action_space, gymnasium.spaces.Discrete):
            self.compute_action = compute_action_discrete
        else:
            self.compute_action = compute_action_box

    def eval_genomes(self, genomes, config):
        self.generation += 1

        jobs = []
        for _, genome in genomes:
            jobs.append(self.pool.apply_async(
                compute_reward,
                [genome, config, self.env, self.compute_action]
            ))

        for job, (_, genome) in zip(jobs, genomes):
            reward = job.get(timeout=None)
            genome.fitness = reward

    # use function to close env when done
    def __delete__(self):
        self.env.close()
        self.pool.close()
        self.pool.join()
        self.pool.terminate()


def run(config_file, env_name, env_args=None, num_generations=None, checkpoint=None):
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config", config_file)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Load the population if checkpoint is not None
    pop = neat.Checkpointer.restore_checkpoint(checkpoint) if checkpoint is not None else neat.Population(config)

    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.Checkpointer(generation_interval=100000, time_interval_seconds=1800))

    ec = ParallelRewardEvaluator(NUM_CORES, env_name, env_args)

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
    run(config_file="config-lander", env_name='LunarLander-v2', env_args={"continuous": False}, num_generations=1e8)
