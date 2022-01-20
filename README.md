# DRL4AUV2TRACK
This is a set of tools developed to train an agen (and multiple agents) to find the optimal path to localize and track a target (and multiple targets).

The Deep Reinforcement Learning (DRL) algorithms implemented are:

- DDPG
- TD3
- SAC

The environment to train the agents is based on the OpenAI Particle

The main objective is to find the optimal path that an autonomous vehicle (e.g. autonomous underwater vehicles (AUV) or autonomous surface vehicles (ASV)) should follow in order to localize and track an underwater target using range-only and single-beacon algorithms. The target estimation algorithms implemented are based on:

- Least Squares (LS)
- Particle Filterse (PF)

An example of a trained agent can be seen below.

<img src="https://github.com/imasmitja/DRL4AUV/blob/main/trained_saca.gif" width="200" height="200"/>
<img src="https://github.com/imasmitja/DRL4AUV/blob/main/trained_sacc.gif" width="200" height="200"/>

## Instructions
Follow the next instructions to set up a Windows computer to run the algorithms.

- conda create -n \<env-name\> python=3.6
- conda activate <env-name>
- conda install git
- conda install -c conda-forge ffmpeg
- pip install git+https://github.com/Kojoley/atari-py.git (optional)
- pip install gym==0.10.0
- pip install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=9.2 -c pytorch
- pip install tensorflow==2.1.0
- git clone https://github.com/openai/gym.git

  cd gym
 
  pip install -e .
  
Test (optional):
 - python examples/agents/random_agent.py

Next (optional as I implemented my own MADDPG):
- git clone https://github.com/openai/baselines.git

  cd baselines
  
  pip install -e .
  
Test (opitonal):
- python baselines/deepq/experiments/train_cartpole.py
- python baselines/deepq/experiments/enjoy_cartpole.py

Next:
- pip install tensorboardX
- pip install imageio
- pip install progressbar
- install pyglet==1.3.2

Train the NN network:
Run in CMD -> python main.py


Then, when the NN is trained you can visualize the polots on tensorBoard by:

Run in CMD -> tensorboard --logdir=./log/ --host=127.0.0.1

Run in web -> http://localhost:6006/


Clean:

remove all files in "model_dir" and "log" folders

Part of this has been obtained from [link](https://arztsamuel.github.io/en/blogs/2018/Gym-and-Baselines-on-Windows.html) and [link](https://knowledge.udacity.com/questions/131475), see them for further information.


## Rewards

It was not straightforward implementation. The algorithm is based on Pytorch, and therefore, it is different from the providecd by the original implementation [MADDPG](https://github.com/openai/maddpg). Nonetheless, finally I could obtain a quite satisfactory result, which is shown below.

Firt 300.000 episodes

![alt-link](https://github.com/imasmitja/MADDPG-AUV/blob/six_agents/rewards.JPG)

After them, I trained the same network again using the saved weights.

![alt-link](https://github.com/imasmitja/MADDPG-AUV/blob/six_agents/rewards_pretrined.JPG)

## Hyperparameters
- BUFFER_SIZE =   int(1e6) # Replay buffer size
- BATCH_SIZE  =   512      # Mini batch size
- GAMMA       =   0.95     # Discount factor
- TAU         =   0.01     # For soft update of target parameters 
- LR_ACTOR    =   1e-3     # Learning rate of the actor
- LR_CRITIC   =   1e-4     # Learning rate of the critic
- WEIGHT_DECAY =  0        # L2 weight decay
- UPDATE_EVERY =  30       # How many steps to take before updating target networks
- UPDATE_TIMES =  20       # Number of times we update the networks
- SEED = 3                 # Seed for random numbers
- parallel_envs = 6        # Number of parallel agents
- num_agents = 6           # Number of agents per environment
- number_of_episodes = 300000
- episode_length = 35
- noise = 0.5              # Amplitude of OU noise
- noise_reduction = 0.999  # Reduction of OU noise per episode

Actor and Critic neural network layers configuration
- in_actor = num_agents*2 + (num_agents-1)*2 + 2+2
- hidden_in_actor = 400
- hidden_out_actor = 200
- out_actor = 2
- in_critic = in_actor * num_agents + out_actor * num_agents
- hidden_in_critic = 700
- hidden_out_critic = 350



