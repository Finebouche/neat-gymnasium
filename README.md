![Illustration of Neat Gymnasium](illustration.jpeg)

# Neat Gymnasium

Neat Gymnasium is a project that utilizes NEAT (NeuroEvolution of Augmenting Topologies) to train agents in various OpenAI Gym environments. The project leverages parallel processing to efficiently evaluate rewards across multiple cores, enhancing performance and scalability.

## Features

- **Parallel Processing**: Utilizes multiple CPU cores to evaluate genomes in parallel, significantly speeding up the training process.
- **Customizable Environments**: Supports various Gym environments and custom configurations.
- **Checkpointing**: Automatically saves progress and allows for restoration from checkpoints.
- **Visualization**: Generates visualizations of neural networks and fitness statistics.

## Requirements

- Python 3.11
- NEAT-Python (@Finebouche branch)
- OpenAI Gym

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Finebouche/neat-gymnasium.git
    cd neat-gymnasium
    ```

2. Install the required packages:
    ```bash
    conda env create -f environment.yml  
    ```
3. Activate the environment:
    ```bash
    conda activate neat_env
    ```
   
## Usage

### Running the Training

To run the training process, execute the `run` function with appropriate arguments:

```python
from multiprocessing import cpu_count
from neat_gymnasium import run

run(
    config_file="config-walker-hardcore",
    env_name='BipedalWalker-v3',
    env_args={"hardcore": True},
    num_generations=100,
    checkpoint=None,
    num_tests=1,
    num_cores=cpu_count(),
)
```

## Configuration

- **config_file**: Path to the NEAT configuration file.
- **env_name**: Name of the Gym environment.
- **env_args**: Additional arguments for the environment.
- **num_generations**: Number of generations to run.
- **checkpoint**: Path to a checkpoint file to resume training.
- **num_tests**: Number of tests to average the fitness.
- **num_cores**: Number of CPU cores to use for parallel processing.

### Checkpointing

The training process automatically saves checkpoints every 1000 generations. To resume from a checkpoint, specify the checkpoint file in the `run` function.

### Visualization

After training, the best genome and training statistics are visualized and saved in the specified directory.

## Example

Here's an example of running the training for the BipedalWalker environment with hardcore mode enabled:

```python
if __name__ == '__main__':
    run(
        config_file="config-walker-hardcore",
        env_name='BipedalWalker-v3',
        env_args={"hardcore": True},
        num_generations=100,
        checkpoint="neat-checkpoint-5286",
        num_tests=1,
        num_cores=cpu_count(),
    )
```

# License

This project is licensed under the MIT License.

# Acknowledgments

Special thanks to the contributors of NEAT-Python and OpenAI Gym for their incredible tools and libraries.

# Ressources 
- https://github.com/google/brain-tokyo-workshop/blob/master/es-clip/es_clip.py
- https://github.com/ShangtongZhang/DistributedES/blob/master/neat-config/BipedalWalker-v2.txt
- https://github.com/sroj/neat-openai-gym/blob/master/config-frozen-lake-neat
- https://arxiv.org/pdf/1712.00006