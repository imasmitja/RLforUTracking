# Deep Reinforcement Learning for Underwater target Tracking (RLforUTracking)
This is a set of tools developed to train an agen (and multiple agents) to find the optimal path to localize and track a target (and multiple targets).

The Deep Reinforcement Learning (RL) algorithms implemented are:

- DDPG
- TD3
- SAC

The environment to train the agents is based on the OpenAI Particle [link](https://github.com/openai/multiagent-particle-envs).

The main objective is to find the optimal path that an autonomous vehicle (e.g. autonomous underwater vehicles (AUV) or autonomous surface vehicles (ASV)) should follow in order to localize and track an underwater target using range-only and single-beacon algorithms. The target estimation algorithms implemented are based on:

- Least Squares (LS)
- Particle Filterse (PF)

An example of a trained agent can be seen below.

| <img src="https://github.com/imasmitja/DRL4AUV/blob/main/trained_saca.gif" width="300" height="300"/> | <img src="https://github.com/imasmitja/DRL4AUV/blob/main/trained_sacc.gif" width="300" height="300"/> |
| --- | --- |

<sup><sub>Legend: Blue dot = agnet, Black dot = target, and Red dot = predicted target position using LS</sup></sub>

## Installation Instructions
Follow the next instructions to set up a Windows computer to run the algorithms.

```
$ conda create -n <env-name> python=3.6
$ conda activate <env-name>
$ conda install git
$ conda install -c conda-forge ffmpeg
$ pip install gym==0.10.0
$ conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=9.2 -c pytorch
$ pip install tensorflow==2.1.0
$ pip install tensorboardX
$ pip install imageio
$ pip install progressbar
$ pip install pyglet==1.3.2
$ pip install cloudpickle
$ pip install tqdm
$ conda install matplotlib
```

Then type `git clone`, and paste the project URL, to clone this repository in your local computer.

```
$ git clone https://github.com/imasmitja/DRLforUTracking
```

## Execution Instructions
Train the DRL network:

```
$ python main.py <configuration file>
```

While the DRL is training you can visualize the polots on tensorBoard by:

```
$ tensorboard --logdir=./log/<configuration file> --host=127.0.0.1
```

Then (Run in web):

```
http://localhost:6006/
```

See a trained agent:

```
$ python see_trained_agent.py <configuration file>
```

Note: `<configuration file>` without extension

An example of the `<configuration file>` can be seen [here](https://github.com/imasmitja/DRL4AUV/blob/main/test_configuration.txt)
  
## Additional information


This repositori is part of the Artificial Intelligence methods for Underwater target Tracking (AIforUTracking) project (ID: 893089) from a Marie Sklodowska-Curie Indvidual Fellowship. More info can be found [here](https://cordis.europa.eu/project/id/893089).

**Acknowledgements** - Anyone using DRLforUTracking data for a publication or project acknowledges and references this [forthcoming] publication.


<img src="https://github.com/imasmitja/DRLforUTracking/blob/main/logos/mscacolor.png" width="235" height="75"/> <img src="https://github.com/imasmitja/DRLforUTracking/blob/main/logos/euflag.png" width="121" height="75"/> <img src="https://github.com/imasmitja/DRLforUTracking/blob/main/logos/logos_poc_aei_2021.jpg" width="393" height="75"/>

<sub><sup>“This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No 893089.”</sup></sub>


**Collaborators**
<img src="https://github.com/imasmitja/DRLforUTracking/blob/main/logos/icm.png" width="100" height="75"/> <img src="https://github.com/imasmitja/DRLforUTracking/blob/main/logos/mbari.png" width="214" height="75"/> <img src="https://github.com/imasmitja/DRLforUTracking/blob/main/logos/udg.png" width="133" height="75"/> <img src="https://github.com/imasmitja/DRLforUTracking/blob/main/logos/upc.jpg" width="331" height="75"/> 
<img src="https://github.com/imasmitja/DRLforUTracking/blob/main/logos/bsc.jpg" width="279" height="75"/>
