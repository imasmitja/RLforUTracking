# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 20:37:13 2021

@author: Usuari

5. Watch a Smart Agent!
In the next code cell, you will load the trained weights from file to watch a smart agent!
"""
import envs
from buffer import ReplayBuffer, ReplayBuffer_SummTree
from maddpg import MADDPG
from masac import MASAC
from matd3_bc import MATD3_BC
import torch
import numpy as np
from tensorboardX import SummaryWriter
import os
from utilities import transpose_list, transpose_to_tensor, circle_path, random_levy
import time
import copy
import matplotlib.pyplot as plt
import pickle

# for saving gif
import imageio

BUFFER_SIZE =   4000 # int(1e6) # Replay buffer size
BATCH_SIZE  =   32 #512      # Mini batch size
GAMMA       =   0.99 #0.95     # Discount factor
TAU         =   0.01     # For soft update of target parameters 
LR_ACTOR    =   1e-3     # Learning rate of the actor
LR_CRITIC   =   1e-3     # Learning rate of the critic
WEIGHT_DECAY =  0 #1e-5     # L2 weight decay
UPDATE_EVERY =  30       # How many steps to take before updating target networks
UPDATE_TIMES =  20       # Number of times we update the networks
SEED =  323 #212309 #21 #60 #1345 #1111 #412   #198                # Seed for random numbers
BENCHMARK   =   True
EXP_REP_BUF =   False     # Experienced replay buffer activation
PRE_TRAINED =   True    # Use a previouse trained network as imput weights
#Scenario used to train the networks
# SCENARIO    =   "simple_track_ivan" 
# SCENARIO    =   "dynamic_track_ivan"
# SCENARIO    =   "dynamic_track_ivan(linear)" 
# SCENARIO    =   "dynamic_track_ivan(random)"
# SCENARIO    =   "dynamic_track_ivan(levy)"
SCENARIO = "tracking"
landmark_depth= 15.
landmark_movable= True
movement= "linear"
pf_method= False
rew_err_th= 0.0003
rew_dis_th= 0.3

RENDER = True #in BSC machines the render doesn't work
PROGRESS_BAR = True #if we want to render the progress bar
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") #To run the pytorch tensors on cuda GPU
RNN = True
HISTORY_LENGTH = 5
# DNN = 'MADDPG'
# DNN = 'MATD3_BC'
DNN = 'MASAC'
START_STEPS = 10000  #Uniform random steps at the begining as suggested by https://spinningup.openai.com/en/latest/algorithms/ddpg.html
REWARD_WINDOWS = 100000 #Sliding windows to measure the avarage reward among epochs
LANDMARK_ERROR_WINDOWS = 10000 #Sliding windows to measure the avarage landmark error among epochs
ALPHA = 0.05
AUTOMATIC_ENTROPY = True
DIM_1 = 64 #it was 64 or 128
DIM_2 = 32 #it was 32 or 128
# DIM_1 = 128 #it was 64 or 128
# DIM_2 = 128 #it was 32 or 128

CIRCLE = False
CIRCLE_RADI = 110.

def seeding(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    global RNN
    seeding(seed = SEED)
    # number of parallel agents
    parallel_envs = 1
    # number of agents per environment
    num_agents = 1
    # number of landmarks (or targets) per environment
    num_landmarks = 1
    #landmark depth
    landmark_depth = 15.
    
    # initialize environment
    torch.set_num_threads(parallel_envs)
    # env = envs.make_parallel_env(parallel_envs, SCENARIO, seed = SEED, num_agents=num_agents, num_landmarks=num_landmarks, landmark_depth=landmark_depth, benchmark = BENCHMARK)
    env = envs.make_parallel_env(parallel_envs, SCENARIO, seed = SEED, num_agents=num_agents, num_landmarks=num_landmarks, landmark_depth=landmark_depth, landmark_movable=landmark_movable, movement=movement, pf_method=pf_method, rew_err_th=rew_err_th, rew_dis_th=rew_dis_th, benchmark = BENCHMARK)
    
    # initialize policy and critic
    if DNN == 'MADDPG':
            maddpg = MADDPG(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn=RNN, dim_1=DIM_1, dim_2=DIM_2)
    elif DNN == 'MATD3_BC':
            maddpg = MATD3_BC(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn=RNN, dim_1=DIM_1, dim_2=DIM_2)
    elif DNN == 'MASAC':
            maddpg =    MASAC(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn = RNN, alpha = ALPHA, automatic_entropy_tuning = AUTOMATIC_ENTROPY, dim_1=DIM_1, dim_2=DIM_2)
    else:
        print('ERROR UNKNOWN DNN ARCHITECTURE')
    agents_reward = []
    for n in range(num_agents):
        agents_reward.append([])
    
    if PRE_TRAINED == True:
     
        # New corrected reward:
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\091321_192609\model_dir\episode-200000.pt' #Test 59, MADDPG
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\091321_202342\model_dir\episode-50000.pt' #Test 59, TD3_BD.
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\091421_070103\model_dir\episode-200000.pt' #Test 68, TD3_BD. From my pc test
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\091521_081505\model_dir\episode-1599992.pt' #Test 68, TD3_BD. From BSC test
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\091521_081505\model_dir\episode-900000.pt' #Test 68, TD3_BD. From BSC test
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\091421_185237\model_dir\episode-100000.pt' #Test 69, TD3_BD. From my pc test
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\091621_092922\model_dir\episode-1599992.pt' #Test 69, TD3_BD. From BSC test
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\091621_092922\model_dir\episode-1500000.pt' #Test 69, TD3_BD. From BSC test
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\091721_124134\model_dir\episode-1599992.pt' #Test 70, MADDPG. From BSC test
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\091721_232551\model_dir\episode-1599992.pt' #Test 702, MADDPG. From BSC test
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\091721_153510\model_dir\episode-1500000.pt' #Test 71, MADDPG. From BSC test history_length = 20
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\091721_171920\model_dir\episode-1450000.pt' #Test 72, MADDPG. From BSC test different reward function
        
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\092921_033654\model_dir\episode-2300000.pt' #Test 94, MADDPG, no LSTM. reward based on distance (1.) but also error (0.01). Batch size=32
        
        ## Test A ##
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\092921_213048\model_dir\episode-2050000.pt' #Test 97, MADDPG, yes LSTM. reward based on distance (1.) but also error (0.01). Batch size dynamic 32*2
        # RNN = True
        
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\092921_215100\model_dir\episode-2200000.pt' #Test 98, MADDPG, no LSTM. reward based on distance (1.) but also error (0.01). Batch size dynamic 32*2
        # RNN = False
        
        ## Test B ##
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\093021_165144\model_dir\episode-2050000.pt' #Test 100, MADDPG, yes LSTM. reward based on distance (0.01) but also error (0.01). Batch size dynamic 32*2
        # RNN = True
        
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\092921_230636\model_dir\episode-2200000.pt' #Test 99, MADDPG, no LSTM. reward based on distance (0.01) but also error (0.01). Batch size dynamic 32*2
        # RNN = False
        
        ## Test C ##
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\093021_182330\model_dir\episode-50000.pt' #Test 101, MADDPG, yes LSTM. (it doesn't leanred)
        # RNN = True
        
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\093021_223141\model_dir\episode-1600000.pt' #Test 102, MADDPG, no LSTM.
        # RNN = False
        
        ## Test D ##
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\093021_234756\model_dir\episode-1550000.pt' #Test 106, MADDPG, yes LSTM. 
        # RNN = True
       
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\100121_155618\model_dir\episode-1650000.pt' #Test 107, MADDPG, no LSTM.
        # RNN = False
        
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\100721_215722\model_dir\episode-1000000.pt' #Test 107, MADDPG, no LSTM.
        # RNN = False
        
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\100721_233716\model_dir\episode-2200000.pt' #Test 116, MADDPG, no LSTM.
        # RNN = False
        
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\101921_154231\model_dir\episode_best.pt' #Test 190 DDPG
        # RNN = False
        
        
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\102121_223057\model_dir\episode' #Test 202 SAC
        # RNN = False
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\102221_153138\model_dir\episode' #Test 211 TD3
        # RNN = False
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\102221_155005\model_dir\episode' #Test 212 SAC
        # RNN = False
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\102221_161316\model_dir\episode' #Test 213 SAC
        # RNN = False
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\102221_224939\model_dir\episode' #Test 223 SAC
        # RNN = False
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\102321_005821\model_dir\episode' #Test 230 TD3
        # RNN = False
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\102321_021705\model_dir\episode' #Test 231 TD3
        # RNN = False
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\102321_164500\model_dir\episode' #Test 232 TD3
        # RNN = False
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\102321_223910\model_dir\episode' #Test 234 TD3
        # RNN = False
        
        # Both algorithms are equal, but the first one using rew7 and the second using rew8
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\102221_161316\model_dir\episode' #Test 213 SAC
        # RNN = False
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\110321_003731_15\model_dir\episode' #Test 282 SAC
        # RNN = False
        
        
        #Agents trained in a dynamic linear target
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\110321_023707_16\model_dir\episode' #Test 290 DDPG
        # RNN = False
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\110321_025137_17\model_dir\episode' #Test 291 TD3
        # RNN = False
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\110321_183336_18\model_dir\episode' #Test 292 MASAC
        # RNN = False
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\110321_183336_19\model_dir\episode' #Test 292 MASAC
        # RNN = False
        
        # MULTI-AGENT
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\111021_030001_1\model_dir\episode' #Test 300 MASAC
        # RNN = False
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\111021_030000_2\model_dir\episode' #Test 301 MATD3
        # RNN = False
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\tm4i\model_dir\episode' #Test 301 SAC auto
        # RNN = False
        
        #Agents trained with rew6(with step) at target depth equal to 15m, 100m, and 200m
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\111521_214647_13\model_dir\episode' #Test 413 SAC 15m
        # RNN = False
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\111721_223501_33\model_dir\episode' #Test 433 SAC 100m
        # RNN = False
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\111821_215406_53\model_dir\episode' #Test 453 SAC 200m
        # RNN = False
        
        #Agents trained with rew1(with step) at target depth equal to 15m using all the algorithms
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\111521_142835_0\model_dir\episode' #Test 400 DDPG
        # RNN = False
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\111521_154106_1\model_dir\episode' #Test 401 TD3
        # RNN = False
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\111621_205031_2\model_dir\episode' #Test 402 SAC 0.005
        # RNN = False
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\111521_181342_3\model_dir\episode' #Test 403 SAC AUTHO
        # RNN = False
        
        #Agents trained with rew2(with step) at step error equal to 0.0003 instead of 0.001 (e.g. T413). Moreover the landmark position is boundered to don't reach out of world and collide at initial position
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\112021_070126_12\model_dir\episode' #Test 702 SAC 
        # RNN = False
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\112021_070126_13\model_dir\episode' #Test 703 SAC AUTHO
        # RNN = False
        
        #Agents trained with rew1 with different steps configuration
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\102121_223057\model_dir\episode' #Test 202 SAC
        # RNN = False
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\111621_205031_2\model_dir\episode' #Test 402 SAC
        # RNN = False
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\111921_215245_2\model_dir\episode' #Test 402b SAC
        # RNN = False
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\112021_161220_2\model_dir\episode' #Test 712 SAC
        # RNN = False
        
        #Agents trained with rew6 with different steps configuration
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\102221_155005\model_dir\episode' #Test 212 SAC
        # RNN = False
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\111521_214647_12\model_dir\episode' #Test 412 SAC
        # RNN = False
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\112021_070126_12\model_dir\episode' #Test 702 SAC 
        # RNN = False
        
        #Agents trained with rew6 with different steps configuration _ Second run for conference paper
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\t6a\model_dir\episode' #Test  
        # RNN = False
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\t7b\model_dir\episode' #Test  
        # RNN = False
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\t7c\model_dir\episode' #Test   
        # RNN = False
        
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\102221_161316\model_dir\episode' #Test 213 SAC AUTHO
        # RNN = False
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\111521_214647_13\model_dir\episode' #Test 413 SAC AUTHO
        # RNN = False
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\112021_070126_13\model_dir\episode' #Test 703 SAC AUTHO
        # RNN = False
        
        # Agent trained at different depths using the new rew6 with steps
        # 15m
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\t8_15mi\model_dir\episode' #Test SAC auto
        # RNN = False
        # 100m
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\t8_100mi\model_dir\episode' #Test SAC auto
        # RNN = False
        # 200m
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\t8_200mi\model_dir\episode' #Test SAC auto
        # RNN = False
        # 300m
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\t8_300mi\model_dir\episode' #Test SAC auto
        # RNN = False
        # using rw6 as before but the error limit to 0.0004 instead of 0.0003
        # 200m
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\t8_200mb\model_dir\episode' #Test SAC auto
        # RNN = False
        # 300m
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\t8_300mb\model_dir\episode' #Test SAC auto
        # RNN = False
        
        #New test using LSTM
        trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\t8b_lstm\model_dir\episode' #Test SAC auto
        RNN = True
        
        aux = torch.load(trained_checkpoint+'_best.pt')
        if DNN == 'MASAC':
            with open(trained_checkpoint +  '_target_entropy_best.file', "rb") as f:
                target_entropy_aux = pickle.load(f)
            with open(trained_checkpoint +  '_log_alpha_best.file', "rb") as f:
                log_alpha_aux = pickle.load(f)
            with open(trained_checkpoint + '_alpha_best.file', "rb") as f:
                alpha_aux = pickle.load(f)
        for i in range(num_agents):  
            if DNN == 'MADDPG':
                maddpg.maddpg_agent[i].actor.load_state_dict(aux[i]['actor_params'])
                maddpg.maddpg_agent[i].critic.load_state_dict(aux[i]['critic_params'])
            elif DNN == 'MATD3_BC':
                maddpg.matd3_bc_agent[i].actor.load_state_dict(aux[i]['actor_params'])
                maddpg.matd3_bc_agent[i].critic.load_state_dict(aux[i]['critic_params'])
            elif DNN == 'MASAC':
                if AUTOMATIC_ENTROPY:
                    maddpg.masac_agent[i].actor.load_state_dict(aux[i]['actor_params'])
                    maddpg.masac_agent[i].critic.load_state_dict(aux[i]['critic_params'])
                    # maddpg.masac_agent[i].target_critic.load_state_dict(aux[i]['target_critic_params'])
                    # maddpg.masac_agent[i].actor_optimizer.load_state_dict(aux[i]['actor_optim_params'])
                    # maddpg.masac_agent[i].critic_optimizer.load_state_dict(aux[i]['critic_optim_params'])
                    # maddpg.masac_agent[i].alpha_optimizer.load_state_dict(aux[i]['alpha_optim_params'])
                    #load agents alpha parameters
                    maddpg.masac_agent[i].target_entropy = target_entropy_aux[i]
                    maddpg.masac_agent[i].log_alpha = log_alpha_aux[i]
                    maddpg.masac_agent[i].alpha = alpha_aux[i]
                else:
                    maddpg.masac_agent[i].actor.load_state_dict(aux[i]['actor_params'])
                    maddpg.masac_agent[i].critic.load_state_dict(aux[i]['critic_params'])
                    # maddpg.masac_agent[i].target_critic.load_state_dict(aux[i]['target_critic_params'])
                    # maddpg.masac_agent[i].actor_optimizer.load_state_dict(aux[i]['actor_optim_params'])
                    # maddpg.masac_agent[i].critic_optimizer.load_state_dict(aux[i]['critic_optim_params'])
            else:
                break
    
    #Reset the environment
    all_obs = env.reset() 
    # flip the first two indices
    obs_roll = np.rollaxis(all_obs,1)
    obs = transpose_list(obs_roll)
    
    #Reset landmark error benchmark
    landmark_error = []
    for i in range(num_landmarks):
        landmark_error.append([])
    landmark_error_episode = []
    for i in range(num_landmarks):
        landmark_error_episode.append([])
    
    #Initialize history buffer with 0.
    obs_size = obs[0][0].size
    history = copy.deepcopy(obs)
    for n in range(parallel_envs):
        for m in range(num_agents):
            for i in range(HISTORY_LENGTH-1):
                if i == 0:
                    history[n][m] = history[n][m].reshape(1,obs_size)*0.
                aux = obs[n][m].reshape(1,obs_size)*0.
                history[n][m] = np.concatenate((history[n][m],aux),axis=0)
    #Initialize action history buffer with 0.
    history_a = np.zeros([parallel_envs,num_agents,HISTORY_LENGTH,1]) #the last entry is the number of actions, here is 2 (x,y)
    
    scores = 0                
    t = 0
    
    #save gif
    frames = []
    gif_folder = ''
    main_folder = trained_checkpoint.split('\\')
    for i in range(len(main_folder)-2):
        gif_folder += main_folder[i]
        gif_folder += '\\'
    total_rewards = []
    steps = []
    agent_x = []
    agent_y = []
    range_total = []
    for i in range(num_agents):
        agent_x.append([])
        agent_y.append([])
        range_total.append([])
    landmark_x = []
    landmark_y = []
    landmark_p_x = []
    landmark_p_y = []
    episodes = 0
    episodes_total = []
    while t<200:
        frames.append(env.render('rgb_array'))
        t +=1
        # select an action
        his = []
        for i in range(num_agents):
            his.append(torch.cat((transpose_to_tensor(history)[i],transpose_to_tensor(history_a)[i]), dim=2))
        # actions = maddpg.act(transpose_to_tensor(obs), noise=0.)       
        # actions = maddpg.act(transpose_to_tensor(history), noise=0.) 
        actions = maddpg.act(his,transpose_to_tensor(obs) , noise=0.0) 
        
        # print('actions=',actions)
         
        actions_array = torch.stack(actions).detach().numpy()
        actions_for_env = np.rollaxis(actions_array,1)
        
        #cirlce path using my previous functions
        if CIRCLE == True:
            actions_for_env = circle_path(obs,CIRCLE_RADI,t) #radius of the desired agent circunference, between 50m and 1000m
        # print('actions=',actions_for_env)
        
        
        # actions_for_env = np.array([[[np.pi*2./10./0.3]]])
        # if t  > 10:
        #     actions_for_env = np.array([[[0.,0.1]]])
        # if t  > 20:
        #     actions_for_env = np.array([[[0.,0.1]]])
        # if t  > 30:
        #     actions_for_env = np.array([[[0.,0.1]]])
        # if t  > 40:
        #     actions_for_env = np.array([[[1.,0.1]]])
        
        #see a random agent
        # actions_for_env = np.array([[np.random.rand(1)*2-1]])
        beta = 1.99 #must be between 1 and 2
        # actions_for_env = random_levy(beta)
        
        # import pdb; pdb.set_trace()
        
        # send all actions to the environment
        next_obs, rewards, dones, info = env.step(actions_for_env)

        # Update history buffers
        # Add obs to the history buffer
        for n in range(parallel_envs):
            for m in range(num_agents):
                aux = obs[n][m].reshape(1,obs_size)
                history[n][m] = np.concatenate((history[n][m],aux),axis=0)
                history[n][m] = np.delete(history[n][m],0,0)
        # Add actions to the history buffer
        history_a = np.concatenate((history_a,actions_for_env.reshape(parallel_envs,num_agents,1,1)),axis=2)
        history_a = np.delete(history_a,0,2)
                    
        # update the score (for each agent)
        scores += np.sum(rewards)  
        # Save values to plot later on
        total_rewards.append(np.sum(rewards))
        steps.append(t)          
        for n in range(parallel_envs):
            for m in range(num_agents):
                agent_x[m].append(obs[n][m][2])
                agent_y[m].append(obs[n][m][3])
                range_total[m].append(obs[n][m][6])
            for mm in range(num_landmarks):
                landmark_x.append(info[0]['n'][0][1][0][0])
                landmark_y.append(info[0]['n'][0][1][0][1])
                landmark_p_x.append(obs[n][m][4]+obs[n][m][2])
                landmark_p_y.append(obs[n][m][5]+obs[n][m][3])
                

        # for e, inf in enumerate(info):
        #     for a in range(num_agents):
        #         agent_info[a] = np.add(agent_info[a],(inf['n'][a]))
                
        # print ('\r\n Rewards at step %i = %.3f'%(t,scores))
        # roll over states to next time step  
        obs = next_obs     

        # print("Score: {}".format(scores))
        episodes += 1
        episodes_total.append(episodes)
        if np.any(dones):
            print('done')
            print('Next:')
            episodes = 0
            #env_wrapper line 18: env.reset(). Therefore, if you don't want an env.reset, comment this line.
            # break
     
    #save .gif
    imageio.mimsave(os.path.join(gif_folder, 'seed-{}.gif'.format(SEED)), 
                                frames, duration=.04)
    
    
    plt.figure(figsize=(5,5))
    plt.plot(steps,total_rewards,'bo-')
    plt.ylabel('Rewards')
    plt.xlabel('Steps')
    plt.title('Trained agent (RL)')
    # plt.title('Predefined cricumference')
    plt.show()
    
    colors = ['royalblue','orangered','gold','seagreen']
    plt.figure(figsize=(5,5))
    for i in range(num_agents):
        agent_xv = np.array(agent_x[i])[:-1]-np.array(agent_x[i])[1:]
        agent_yv = np.array(agent_y[i])[:-1]-np.array(agent_y[i])[1:]
        agent_v = np.sqrt(agent_xv**2 + agent_yv**2)
        plt.plot(steps[1:],agent_v,'bo--', color=colors[i], alpha=0.5,label='Agent')
    landmark_xv = np.array(landmark_x)[:-1]-np.array(landmark_x)[1:]
    landmark_yv = np.array(landmark_y)[:-1]-np.array(landmark_y)[1:]
    landmark_v = np.sqrt(landmark_xv**2 + landmark_yv**2)
    plt.plot(steps[1:],landmark_v,'k^--',alpha=0.5,label='Landmark')
    plt.ylabel('relative velocity')
    plt.xlabel('Steps')
    plt.title('Trained agent (RL)')
    # plt.title('Predefined cricumference')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(5,5))
    for i in range(num_agents):
        plt.plot(agent_x[i],agent_y[i],'bo--', color=colors[i], alpha=0.5,label='Agent')
    plt.plot(landmark_p_x[4:],landmark_p_y[4:],'rs--',color='orangered',alpha=0.5,label='Landmark Predicted')
    plt.plot(landmark_x,landmark_y,'k^--',alpha=0.5,label='Landmark Real')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    # plt.title('Test 2b')
    plt.axis('equal')
    # plt.xlim(0.26,0.3)
    # plt.ylim(-0.14,-0.08)
    
    # plt.xlim(0.4,1.1)
    # plt.ylim(-0.9,-0.3)
    leg = plt.legend(loc='lower left')
    leg.get_frame().set_edgecolor('w')
    plt.tick_params(direction='in',bottom=True,top=True,left=True,right=True,which='major',width = 0.75, length=2.5)
    plt.tick_params(direction='in',bottom=True,top=True,left=True,right=True,which='minor',width = 0.5, length=1.5)
    plt.grid(which='major', linestyle='-', linewidth='.8', alpha=0.4)
    plt.grid(which='minor', linestyle='-', linewidth='.4', alpha=0.4)
    
    # plt.title('Predefined cricumference')
    plt.savefig('depth_test.png',format='png', dpi=800 ,bbox_inches='tight',pad_inches = 0.02)
    plt.show()
    
    target_error = np.sqrt((np.array(landmark_p_x)-np.array(landmark_x))**2+(np.array(landmark_p_y)-np.array(landmark_y))**2)
    plt.figure(figsize=(5,5))
    plt.plot(steps,target_error,'bo-')
    plt.hlines(0.0003,0,210, colors='k', linestyles='--')
    plt.ylabel('Target prediction error (RMSE)')
    plt.xlabel('Steps')
    plt.title('Trained agent (RL)')
    plt.ylim(0,0.002)
    # plt.title('Predefined cricumference')
    plt.show()
    print('RMSE= %.3f m; STD = %.3f m' % (np.mean(target_error[-100:])*1000., np.std(target_error[-100:])*1000.))
    
    plt.figure(figsize=(5,5))
    for i in range(num_agents):
        plt.plot(steps,range_total[i],'bo-')
    plt.ylabel('Range')
    plt.xlabel('Steps')
    plt.title('Trained agent (RL)')
    plt.grid()
    plt.ylim(0,0.5)
    # plt.title('Predefined cricumference')
    plt.show()
    print('avg range = %.1f m; STD = %.3f m'% (np.mean(range_total[0][-100:])*1000., np.std(range_total[0][-100:])*1000.))
    
    plt.figure(figsize=(5,5))
    plt.plot(steps,episodes_total,'bo-')
    plt.ylabel('Number of episodes')
    plt.xlabel('Steps')
    plt.title('Trained agent (RL)')
    # plt.title('Predefined cricumference')
    plt.show()
    
    print('MEAN SCORE = ',scores)
    print('TOTAL LAST SCORE = ',np.mean(total_rewards[::-1][:10]))
    
    while True:
        a = 0
        break
    
    env.close()
    
if __name__=='__main__':
    main()
    