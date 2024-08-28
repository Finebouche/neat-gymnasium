# based on reporting.py and statistics.py, make a WandbReporter
import wandb
from neat.reporting import BaseReporter
import numpy as np
import gymnasium
import neat

class WandbReporter(BaseReporter):
    def __init__(self, api_key, project_name, env, generation_interval, tags=None):
        super().__init__()
        self.api_key = api_key
        self.env = env
        self.project_name = project_name
        self.tags = tags
        self.generation_interval = generation_interval
        self.current_generation = None

    def start_generation(self, generation):
        self.current_generation = generation
        wandb.init(project=self.project_name, tags=self.tags)
        wandb.log({"generation": generation})

    def end_generation(self, config, population, species_set):
        pass

    def post_evaluate(self, config, population, species, best_genome):
        wandb.log({"best_genome": best_genome.fitness})
        net = neat.nn.FeedForwardNetwork.create(best_genome, config)
        # save the video every generation_interval
        if self.generation_interval is not None and self.current_generation % self.generation_interval == 0:
            observation, observation_init_info = self.env.reset()
            frames = []
            while True:
                if isinstance(self.env.action_space, gymnasium.spaces.Discrete):
                    action = np.argmax(net.activate(observation))
                else:
                    action = net.activate(observation)
                observation, _,  terminated, done, _ = self.env.step(action)
                frame = self.env.render()
                frames.append(frame.transpose(2, 0, 1))  # Collect frame
                print(frame.T.shape)
                if terminated or done:
                    break

            numpy_array_video = np.array(frames)
            print("numpy_array_video", numpy_array_video.shape)
            wandb.log({"video": wandb.Video(numpy_array_video, fps=4, format="gif")})

    def post_reproduction(self, config, population, species):
        pass

    def complete_extinction(self):
        pass

    def found_solution(self, config, generation, best):
        pass

    def species_stagnant(self, sid, species):
        pass

    def info(self, msg):
        pass
