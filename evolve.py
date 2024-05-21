from multiprocessing import Pool, cpu_count
import os
import pickle
import neat
import visualize
from compute_action_util import compute_reward


class ParallelRewardEvaluator(object):
    def __init__(self, num_workers, env_name, env_args, penalize_inactivity, num_tests):
        self.num_workers = num_workers
        self.generation = 0
        self.env_name = env_name
        self.env_args = env_args
        self.penalize_inactivity = penalize_inactivity
        self.num_tests = num_tests
        self.num_workers = num_workers
        self.pool = Pool(processes=num_workers)

    def eval_genomes(self, genomes, config):
        self.generation += 1
        if self.num_workers < 2:
            for genome_id, genome in genomes:
                genome.fitness = compute_reward(genome, config, self.env_name, self.env_args, self.penalize_inactivity, self.num_tests)
            return
        else:
            jobs = []
            for _, genome in genomes:
                jobs.append(self.pool.apply_async(
                    compute_reward,
                    [genome, config, self.env_name, self.env_args, self.penalize_inactivity, self.num_tests]
                ))

            for job, (_, genome) in zip(jobs, genomes):
                reward = job.get(timeout=None)
                genome.fitness = reward

    # use function to close env when done
    def __delete__(self):
        self.pool.close()
        self.pool.close()
        self.pool.join()
        self.pool.terminate()


def run(config_file, env_name, env_args=None, penalize_inactivity=False, num_generations=None, checkpoint=None, num_tests=5, num_cores=1):
    print("Charging environment : ", env_name)
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
    pop.add_reporter(neat.Checkpointer(generation_interval=1000, time_interval_seconds=1800))

    ec = ParallelRewardEvaluator(num_cores, env_name, env_args, penalize_inactivity, num_tests)

    # Run until the winner from a generation is able to solve the environment
    gen_best = pop.run(ec.eval_genomes, num_generations)

    # get strings name of env_args which value are True
    env_args_str = [key for key, value in env_args.items() if value]
    result_path = os.path.join(local_dir, "visualisations", env_name, *env_args_str)
    # Save the best model
    # create the directory if it does not exist
    os.makedirs(result_path, exist_ok=True)
    with open(result_path + '/best_genome.pickle', 'wb') as f:
        pickle.dump(gen_best, f)

    # Display the winning genome.
    visualize.draw_net(config, gen_best, view=False, filename=result_path + "/win-net.gv")
    visualize.plot_stats(stats, ylog=False, view=False, filename=result_path + "/avg_fitness.svg")


if __name__ == '__main__':
    # https://github.com/ShangtongZhang/DistributedES/blob/master/neat-config/BipedalWalker-v2.txt
    # LunarLander-v2 CarRacing-v1, BipedalWalker-v3, CartPole-v1
    run(config_file="config-walker-hardcore",
        env_name='BipedalWalker-v3',
        env_args={"hardcore": True},  # "continuous": False, "hardcore": True
        penalize_inactivity=True,
        num_generations=1e4,
        checkpoint=None,
        num_tests=1,
        num_cores=cpu_count(),
    )
