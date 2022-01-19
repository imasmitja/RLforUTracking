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
from matd3_bc import MATD3_BC
from masac import MASAC
import torch
import numpy as np
from tensorboardX import SummaryWriter
import os
from utilities import transpose_list, transpose_to_tensor, circle_path, random_levy
import time
import copy
import matplotlib.pyplot as plt
import random
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
SEED = 323 #3   #198                # Seed for random numbers
BENCHMARK   =   True
EXP_REP_BUF =   False     # Experienced replay buffer activation
PRE_TRAINED =   True    # Use a previouse trained network as imput weights
DIM_1 = 64
DIM_2 = 32
#Scenario used to train the networks
# SCENARIO    =   "simple_track_ivan" 
# SCENARIO    =   "dynamic_track_ivan" 
# SCENARIOS = ["simple_track_ivan" ,"dynamic_track_ivan(linear)","dynamic_track_ivan(random)" ,"dynamic_track_ivan(levy)"]
# SCENARIOS = ["dynamic_track_ivan(random)" ,"dynamic_track_ivan(levy)"]
# SCENARIOS = ["simple_track_ivan","dynamic_track_ivan(linear)","dynamic_track_ivan(levy)"]
# SCENARIOS = ["dynamic_track_ivan(linear)"]
# SCENARIOS = ["dynamic_track_ivan(levy)"]
# SCENARIOS = ["simple_track_ivan"]
SCENARIOS = ["tracking"]
landmark_depth= 15.
landmark_movable= False
movement= "linear"
pf_method= False
rew_err_th= 0.0003
rew_dis_th= 0.3

RENDER = False #in BSC machines the render doesn't work
PROGRESS_BAR = True #if we want to render the progress bar
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") #To run the pytorch tensors on cuda GPU
RNN = True
HISTORY_LENGTH = 5
# DNN = 'MADDPG'
# DNN = 'MATD3_BC'
# DNNS = ['MATD3_BC_T68','MATD3_BC_T69','circumference']
# DNNS = ['MATD3_BC_T68','MATD3_BC_T69','circumference','MADDPG_T70','MADDPG_T702','MADDPG_T71','MADDPG_T72']
# DNNS = ['MADDPG_T75','MADDPG_LSTM_T77','circumference','MATD3_LSTM'] #Good set of trined agents with rewards based on distance
# DNNS = ['MADDPG_T98','MADDPG_LSTM_T97','MADDPG_T99','MADDPG_LSTM_T100','MADDPG_T102','MADDPG_LSTM_T101','MADDPG_T107','MADDPG_LSTM_T106','circumference'] #Good set of trined agents with rewards based on distance and prediction error
# DNNS = ['MADDPG_T75','MADDPG_T98','MADDPG_T99','MADDPG_T102','MADDPG_T107','circumference']
# DNNS = ['MADDPG_T200','MATD3_T201','MASAC_T202','MASAC_T213','MASAC_T223','circumference']
# DNNS = ['MADDPG_T290','MATD3_T291','MASAC_T292','MASAC_T293']


# DNNS = ['MADDPG_t5c','MATD3_t6c','MASACc_t7c','MASACa_t8c'] #from presentation OSM
DNNS = ['circumference']


# DNNS = ['MASACc_T212','MASACc_T412','MASACc_T702','circumference']
# DNNS = ['MASACa_T213','MASACa_T413','MASACa_T703','circumference']
# DNNS = ['MASACc_t7c','MASACa_t8c','circumference']
# DNNS = ['MASAC_t8b_lstm']
# DNNS = ['MASACa_t8c']


# DNNS = ['circumference']
NUM_RUNS_MEAN = 100 #1000


NAME_FOLDER = 'E:\\Ivan\\UPC\\GitHub\\plots\\'
NAME_SUBFOLDER = 'LSTM4Trev3_OSM'

os.makedirs(NAME_FOLDER+NAME_SUBFOLDER, exist_ok=True)

def seeding(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    global HISTORY_LENGTH
    global RNN 
    # Run a bunch of tests to compere the results
    steps = []
    agent_x = []
    agent_y = []
    landmark_x = []
    landmark_y = []
    landmark_p_x = []
    landmark_p_y = []
    range_total = []
    total_rewards = []
    for scenario_num, SCENARIO in enumerate(SCENARIOS):
        print('scenario number: %d, name: %s'%(scenario_num,SCENARIO))
        steps.append([])
        agent_x.append([])
        agent_y.append([])
        landmark_x.append([])
        landmark_y.append([])
        landmark_p_x.append([])
        landmark_p_y.append([])
        range_total.append([])
        total_rewards.append([])
        for dnn_num, DNN in enumerate(DNNS):
            print('DNN number: %d, name: %s'%(dnn_num,DNN))
            steps[scenario_num].append([])
            agent_x[scenario_num].append([])
            agent_y[scenario_num].append([])
            landmark_x[scenario_num].append([])
            landmark_y[scenario_num].append([])
            landmark_p_x[scenario_num].append([])
            landmark_p_y[scenario_num].append([])
            range_total[scenario_num].append([])
            total_rewards[scenario_num].append([])
            
            #Initialize seed
            seeding(seed = SEED)
            scores_total = []
            for num_run in range(NUM_RUNS_MEAN):
                steps[scenario_num][dnn_num].append([])
                agent_x[scenario_num][dnn_num].append([])
                agent_y[scenario_num][dnn_num].append([])
                landmark_x[scenario_num][dnn_num].append([])
                landmark_y[scenario_num][dnn_num].append([])
                landmark_p_x[scenario_num][dnn_num].append([])
                landmark_p_y[scenario_num][dnn_num].append([])
                range_total[scenario_num][dnn_num].append([])
                total_rewards[scenario_num][dnn_num].append([])
                
                # number of parallel agents
                parallel_envs = 1
                # number of agents per environment
                num_agents = 1
                # number of landmarks (or targets) per environment
                num_landmarks = 1
                
                # initialize environment
                torch.set_num_threads(parallel_envs)
                # env = envs.make_parallel_env(parallel_envs, SCENARIO, seed = SEED*(num_run), num_agents=num_agents, num_landmarks=num_landmarks, benchmark = BENCHMARK)
                env = envs.make_parallel_env(parallel_envs, SCENARIO, seed = SEED*(num_run), num_agents=num_agents, num_landmarks=num_landmarks, landmark_depth=landmark_depth, landmark_movable=landmark_movable, movement=movement, pf_method=pf_method, rew_err_th=rew_err_th, rew_dis_th=rew_dis_th, benchmark = BENCHMARK)
    
                # agents_reward = []
                # for n in range(num_agents):
                #     agents_reward.append([])
                # initialize policy and critic
                if DNN == 'MADDPG':
                        maddpg = MADDPG(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn=RNN)
                        
                elif DNN == 'MATD3_BC_T68':
                        maddpg = MATD3_BC(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn=RNN)
                        trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\091521_081505\model_dir\episode-900000.pt' #Test 68, TD3_BD. From BSC test
                
                elif DNN == 'MATD3_BC_T69':
                        maddpg = MATD3_BC(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn=RNN)
                        trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\091621_092922\model_dir\episode-1500000.pt' #Test 69, TD3_BD. From BSC test
                
                elif DNN == 'MADDPG_T70':
                        maddpg = MADDPG(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn=RNN)
                        trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\091721_124134\model_dir\episode-1599992.pt' #Test 70, MADDPG. From BSC test
                        
                elif DNN == 'MADDPG_T702':
                        maddpg = MADDPG(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn=RNN)
                        trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\091721_232551\model_dir\episode-1599992.pt' #Test 702, MADDPG. From BSC test
                        
                elif DNN == 'MADDPG_T71':
                        HISTORY_LENGTH = 20
                        maddpg = MADDPG(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn=RNN)
                        trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\091721_153510\model_dir\episode-1500000.pt' #Test 71, MADDPG. From BSC test history_length = 20
                        
                elif DNN == 'MADDPG_T72':
                        maddpg = MADDPG(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn=RNN)
                        trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\091721_171920\model_dir\episode-1450000.pt' #Test 72, MADDPG. From BSC test different reward function
                
                
                elif DNN == 'MADDPG_T75':
                        RNN = False
                        maddpg = MADDPG(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn=RNN)
                        trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\092221_162809\model_dir\episode-3599992.pt' #Test 
                elif DNN == 'MADDPG_LSTM_T77':
                        RNN = True
                        maddpg = MADDPG(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn=RNN)
                        trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\092221_155202\model_dir\episode-2800000.pt' #Test 
                elif DNN == 'MATD3_LSTM':
                        RNN = True
                        maddpg = MATD3_BC(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn=RNN)
                        trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\092321_065219\model_dir\episode-3599992.pt' #Test 
                
                
                
                #TestA
                elif DNN == 'MADDPG_LSTM_T97':
                        RNN = True
                        maddpg = MADDPG(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn=RNN)
                        trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\092921_213048\model_dir\episode-2050000.pt' #Test 97 
                elif DNN == 'MADDPG_T98':
                        RNN = False
                        maddpg = MADDPG(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn=RNN)
                        trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\092921_215100\model_dir\episode-2200000.pt' #Test 98    
                #TestB
                elif DNN == 'MADDPG_LSTM_T100':
                        RNN = True
                        maddpg = MADDPG(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn=RNN)
                        trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\093021_165144\model_dir\episode-2050000.pt' #Test 100 
                elif DNN == 'MADDPG_T99':
                        RNN = False
                        maddpg = MADDPG(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn=RNN)
                        trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\092921_230636\model_dir\episode-2200000.pt' #Test 99,  
                #TestC
                elif DNN == 'MADDPG_LSTM_T101':
                        RNN = True
                        maddpg = MADDPG(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn=RNN)
                        trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\093021_182330\model_dir\episode-50000.pt' #Test 101 
                elif DNN == 'MADDPG_T102':
                        RNN = False
                        maddpg = MADDPG(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn=RNN)
                        trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\093021_223141\model_dir\episode-1600000.pt' #Test 102 
                #TestD
                elif DNN == 'MADDPG_LSTM_T106':
                        RNN = True
                        maddpg = MADDPG(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn=RNN)
                        trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\093021_234756\model_dir\episode-1550000.pt' #Test 106 
    
                elif DNN == 'MADDPG_T107':
                        RNN = False
                        maddpg = MADDPG(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn=RNN)
                        trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\100121_155618\model_dir\episode-1650000.pt' #Test 107,  
                    
                
                
                
                
                #New Set of tests 10/27/2021
                elif DNN == 'MADDPG_T200':
                        RNN = False
                        maddpg = MADDPG(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn=RNN)
                        trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\102121_211303\model_dir\episode' #Test
                elif DNN == 'MATD3_T201':
                        RNN = False
                        maddpg = MATD3_BC(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn=RNN)
                        trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\102321_231429\model_dir\episode' #Test
                elif DNN == 'MASAC_T202':
                        RNN = False
                        ALPHA = 0.005
                        AUTOMATIC_ENTROPY = False
                        maddpg = MASAC(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn = RNN, alpha = ALPHA, automatic_entropy_tuning = AUTOMATIC_ENTROPY)
                        trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\102121_223057\model_dir\episode' #Test
                elif DNN == 'MASAC_T213':
                        RNN = False
                        ALPHA = 0.05
                        AUTOMATIC_ENTROPY = True
                        maddpg = MASAC(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn = RNN, alpha = ALPHA, automatic_entropy_tuning = AUTOMATIC_ENTROPY)
                        trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\102521_031811\model_dir\episode' #Test  
                elif DNN == 'MASAC_T223':
                        RNN = False
                        ALPHA = 0.05
                        AUTOMATIC_ENTROPY = True
                        maddpg = MASAC(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn = RNN, alpha = ALPHA, automatic_entropy_tuning = AUTOMATIC_ENTROPY)
                        trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\102521_232356\model_dir\episode' #Test  
                elif  DNN == 'circumference':
                        RNN = False
                        maddpg = MATD3_BC(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn=RNN)
                        trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\102321_231429\model_dir\episode' #Test 
                
                #New set of tests with dynamic_linear target 11/9/2021
                elif DNN == 'MADDPG_T290':
                        RNN = False
                        maddpg = MADDPG(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn=RNN)
                        trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\110321_023707_16\model_dir\episode' #Test
                elif DNN == 'MATD3_T291':
                        RNN = False
                        maddpg = MATD3_BC(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn=RNN)
                        trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\110321_025137_17\model_dir\episode' #Test
                elif DNN == 'MASAC_T292':
                        RNN = False
                        ALPHA = 0.005
                        AUTOMATIC_ENTROPY = False
                        maddpg = MASAC(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn = RNN, alpha = ALPHA, automatic_entropy_tuning = AUTOMATIC_ENTROPY)
                        trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\110321_183336_18\model_dir\episode' #Test
                elif DNN == 'MASAC_T293':
                        RNN = False
                        ALPHA = 0.05
                        AUTOMATIC_ENTROPY = True
                        maddpg = MASAC(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn = RNN, alpha = ALPHA, automatic_entropy_tuning = AUTOMATIC_ENTROPY)
                        trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\110321_183336_19\model_dir\episode' #Test  
                elif  DNN == 'circumference':
                        RNN = False
                        maddpg = MATD3_BC(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn=RNN)
                        trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\102321_231429\model_dir\episode' #Test 
                
                #Tests conducted for conference paper 1.
                elif DNN == 'MASACc_T212':
                        RNN = False
                        ALPHA = 0.05
                        AUTOMATIC_ENTROPY = False
                        maddpg = MASAC(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn = RNN, alpha = ALPHA, automatic_entropy_tuning = AUTOMATIC_ENTROPY)
                        trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\102221_155005\model_dir\episode' #Test
                elif DNN == 'MASACc_T412':
                        RNN = False
                        ALPHA = 0.05
                        AUTOMATIC_ENTROPY = False
                        maddpg = MASAC(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn = RNN, alpha = ALPHA, automatic_entropy_tuning = AUTOMATIC_ENTROPY)
                        trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\111521_214647_12\model_dir\episode' #Test
                elif DNN == 'MASACc_T702':
                        RNN = False
                        ALPHA = 0.05
                        AUTOMATIC_ENTROPY = False
                        maddpg = MASAC(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn = RNN, alpha = ALPHA, automatic_entropy_tuning = AUTOMATIC_ENTROPY)
                        trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\112021_070126_12\model_dir\episode' #Test
                
                elif DNN == 'MASACa_T213':
                        RNN = False
                        ALPHA = 0.05
                        AUTOMATIC_ENTROPY = True
                        maddpg = MASAC(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn = RNN, alpha = ALPHA, automatic_entropy_tuning = AUTOMATIC_ENTROPY)
                        trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\102221_161316\model_dir\episode' #Test
                elif DNN == 'MASACa_T413':
                        RNN = False
                        ALPHA = 0.05
                        AUTOMATIC_ENTROPY = True
                        maddpg = MASAC(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn = RNN, alpha = ALPHA, automatic_entropy_tuning = AUTOMATIC_ENTROPY)
                        trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\111521_214647_13\model_dir\episode' #Test
                elif DNN == 'MASACa_T703':
                        RNN = False
                        ALPHA = 0.05
                        AUTOMATIC_ENTROPY = True
                        maddpg = MASAC(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn = RNN, alpha = ALPHA, automatic_entropy_tuning = AUTOMATIC_ENTROPY)
                        trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\112021_070126_13\model_dir\episode' #Test
                        
                elif DNN == 'MASACc_t7c':
                        RNN = False
                        ALPHA = 0.05
                        AUTOMATIC_ENTROPY = False
                        maddpg = MASAC(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn = RNN, alpha = ALPHA, automatic_entropy_tuning = AUTOMATIC_ENTROPY)
                        trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\t7c\model_dir\episode' #Test
                elif DNN == 'MASACa_t8c':
                        RNN = False
                        ALPHA = 0.005
                        AUTOMATIC_ENTROPY = True
                        maddpg = MASAC(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn = RNN, alpha = ALPHA, automatic_entropy_tuning = AUTOMATIC_ENTROPY)
                        trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\t8c\model_dir\episode' #Test
                        
                elif DNN == 'MASAC_t8b_lstm':
                        RNN = True
                        ALPHA = 0.005
                        AUTOMATIC_ENTROPY = True
                        maddpg = MASAC(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn = RNN, alpha = ALPHA, automatic_entropy_tuning = AUTOMATIC_ENTROPY, dim_1=DIM_1, dim_2=DIM_2)
                        trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\t8b_lstm\model_dir\episode' #Test
                        
                        
                #for OSM presentation
                elif DNN == 'MADDPG_t5c':
                        RNN = False
                        maddpg = MADDPG(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn=RNN)
                        trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\t5c\model_dir\episode' #Test
                elif DNN == 'MATD3_t6c':
                        RNN = False
                        maddpg = MATD3_BC(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn=RNN)
                        trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\t6c\model_dir\episode' #Test
                elif DNN == 'MASACc_t7c':
                        RNN = False
                        ALPHA = 0.005
                        AUTOMATIC_ENTROPY = False
                        maddpg = MASAC(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn = RNN, alpha = ALPHA, automatic_entropy_tuning = AUTOMATIC_ENTROPY)
                        trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\t7c\model_dir\episode' #Test
                elif DNN == 'MASACa_t8c':
                        RNN = False
                        ALPHA = 0.05
                        AUTOMATIC_ENTROPY = True
                        maddpg = MASAC(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn = RNN, alpha = ALPHA, automatic_entropy_tuning = AUTOMATIC_ENTROPY)
                        trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\t8c\model_dir\episode' #Test
                        
                        
                                                
                elif  DNN == 'circumference':
                        RNN = False
                        maddpg = MATD3_BC(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn=RNN)
                        trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\102321_231429\model_dir\episode' #Test 
                
                
                
                else:
                    print('ERROR UNKNOWN DNN ARCHITECTURE')
    
                aux = torch.load(trained_checkpoint+'_best.pt')
                if DNN.find('MASAC') == 0:
                    with open(trained_checkpoint +  '_target_entropy_best.file', "rb") as f:
                        target_entropy_aux = pickle.load(f)
                    with open(trained_checkpoint +  '_log_alpha_best.file', "rb") as f:
                        log_alpha_aux = pickle.load(f)
                    with open(trained_checkpoint + '_alpha_best.file', "rb") as f:
                        alpha_aux = pickle.load(f)
                for i in range(num_agents):  
                    if DNN.find('MADDPG') == 0:
                        maddpg.maddpg_agent[i].actor.load_state_dict(aux[i]['actor_params'])
                        maddpg.maddpg_agent[i].critic.load_state_dict(aux[i]['critic_params'])
                    elif DNN.find('MATD3') == 0:
                        maddpg.matd3_bc_agent[i].actor.load_state_dict(aux[i]['actor_params'])
                        maddpg.matd3_bc_agent[i].critic.load_state_dict(aux[i]['critic_params'])
                    elif DNN.find('MASAC') == 0:
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
                    elif DNN == 'circumference':
                        maddpg.matd3_bc_agent[i].actor.load_state_dict(aux[i]['actor_params'])
                        maddpg.matd3_bc_agent[i].critic.load_state_dict(aux[i]['critic_params'])
                    else:
                        break
                
                #Reset the environment
                all_obs = env.reset() 
                # flip the first two indices
                obs_roll = np.rollaxis(all_obs,1)
                obs = transpose_list(obs_roll)
                
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
                episodes = 0
                episodes_total = []
                while t<200:
                    if RENDER == True:
                        frames.append(env.render('rgb_array'))
                    t +=1
                    # select an action
                    his = []
                    for i in range(num_agents):
                        his.append(torch.cat((transpose_to_tensor(history)[i],transpose_to_tensor(history_a)[i]), dim=2))
                    # actions = maddpg.act(transpose_to_tensor(obs), noise=0.)       
                    # actions = maddpg.act(transpose_to_tensor(history), noise=0.) 
                    actions = maddpg.act(his,transpose_to_tensor(obs) , noise=0.0) 

                    actions_array = torch.stack(actions).detach().numpy()
                    actions_for_env = np.rollaxis(actions_array,1)
                    
                    #cirlce path using my previous functions
                    if DNN == 'circumference':
                        actions_for_env = circle_path(obs,110.,t) #if this value is bigger, the circle radius is smaller 60 => radi = 200m
                    
                    # print('actions=',actions_for_env)
                    
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
                    # import pdb; pdb.set_trace()
                    total_rewards[scenario_num][dnn_num][num_run].append(np.sum(rewards))
                    steps[scenario_num][dnn_num][num_run].append(t)          
                    for n in range(parallel_envs):
                        for m in range(num_agents):
                            agent_x[scenario_num][dnn_num][num_run].append(obs[n][m][2])
                            agent_y[scenario_num][dnn_num][num_run].append(obs[n][m][3])
                            range_total[scenario_num][dnn_num][num_run].append(obs[n][m][6])
                        for mm in range(num_landmarks):
                            landmark_x[scenario_num][dnn_num][num_run].append(info[0]['n'][0][1][0][0])
                            landmark_y[scenario_num][dnn_num][num_run].append(info[0]['n'][0][1][0][1])
                            landmark_p_x[scenario_num][dnn_num][num_run].append(obs[n][m][4]+obs[n][m][2])
                            landmark_p_y[scenario_num][dnn_num][num_run].append(obs[n][m][5]+obs[n][m][3])
                            
                            
                    # print ('\r\n Rewards at step %i = %.3f'%(t,scores))
                    # roll over states to next time step  
                    obs = next_obs     
            
                    # print("Score: {}".format(scores))
                    episodes += 1
                    episodes_total.append(episodes)
                    if np.any(dones):
                        pass
                        # print('done state reached')
                        # print('Next:')
                        # episodes = 0
                        # break
                env.close()
                scores_total.append(scores)
                print('Iteration number: %d/%d Score obtained: %.3f'%(num_run,NUM_RUNS_MEAN,scores))
                # save_data(steps[scenario_num][dnn_num][num_run],agent_x[scenario_num][dnn_num][num_run],agent_y,landmark_x,landmark_y,landmark_p_x,landmark_p_y,range_total,total_rewards)
            save_data(steps[scenario_num][dnn_num],agent_x[scenario_num][dnn_num],agent_y[scenario_num][dnn_num],landmark_x[scenario_num][dnn_num],landmark_y[scenario_num][dnn_num],landmark_p_x[scenario_num][dnn_num],landmark_p_y[scenario_num][dnn_num],range_total[scenario_num][dnn_num],total_rewards[scenario_num][dnn_num],scenario_num,dnn_num)
            # print('Average score = ', np.mean(scores_total))
            # print('scores = ',scores_total)
    return steps,agent_x,agent_y,landmark_x,landmark_y,landmark_p_x,landmark_p_y,range_total,total_rewards

#%%  
def plot_test(steps,agent_x,agent_y,landmark_x,landmark_y,landmark_p_x,landmark_p_y,range_total,total_rewards):  
    color = colors(len(steps[0]))
    for i, SCENARIO in enumerate(SCENARIOS):        
        # plt.figure(figsize=(5,5))
        # for n, dnn in enumerate(DNNS):
        #     step_mean = np.mean(np.array(steps[i][n]),axis=0)
        #     reward_mean = np.mean(np.array(total_rewards[i][n]),axis=0)
        #     reward_std = np.std(np.array(total_rewards[i][n]),axis=0)
        #     plt.plot(step_mean,reward_mean,'bo-', c=color[n] ,label=dnn)
        #     plt.fill_between(step_mean, reward_mean+reward_std, reward_mean-reward_std, facecolor=color[n], alpha=0.5)
        # plt.legend()
        # plt.ylabel('Rewards')
        # plt.xlabel('Steps')
        # plt.title(SCENARIO)
        # plt.show()
        
        for n, dnn in enumerate(DNNS):
            plt.figure(figsize=(5,5))
            plt.plot(agent_x[i][n][0],agent_y[i][n][0],'bo--',alpha=0.5,label='Agent')
            plt.plot(landmark_p_x[i][n][0],landmark_p_y[i][n][0],'rs--',alpha=0.5,label='Landmark Predicted')
            plt.plot(landmark_x[i][n][0],landmark_y[i][n][0],'k^--',alpha=0.5,label='Landmark Real')
            plt.xlabel('X position')
            plt.ylabel('Y position')
            plt.title(SCENARIO+'-('+dnn+')')
            plt.axis('equal')
            plt.grid()
            plt.xlim(-1,1)
            plt.ylim(-1,1)
            plt.legend()
            plt.show()
        
        target_error = np.sqrt((np.array(landmark_p_x)-np.array(landmark_x))**2+(np.array(landmark_p_y)-np.array(landmark_y))**2)
        plt.figure(figsize=(5,5))
        for n, dnn in enumerate(DNNS):
            step_mean = np.mean(np.array(steps[i][n]),axis=0)
            error_mean = np.mean(np.array(target_error[i][n]),axis=0)
            error_std = np.std(np.array(target_error[i][n]),axis=0)
            # import pdb; pdb.set_trace()
            plt.plot(step_mean,error_mean,'b-', lw=2, c=color[n] ,label=dnn)
            plt.fill_between(step_mean, error_mean+error_std, error_mean-error_std, facecolor=color[n], alpha=0.5)
        plt.legend()
        plt.ylabel('Target prediction error (RMSE)')
        plt.xlabel('Steps')
        plt.title(SCENARIO)
        plt.ylim(0,0.3)
        plt.xlim(0,len(steps[i][n][0]))
        # plt.title('Predefined cricumference')
        plt.show()
        
    
    
    # print('MEAN SCORE = ',scores)
    # print('TOTAL LAST SCORE = ',np.mean(total_rewards[::-1][:10]))
    
    # while True:
    #     a = 0
    # imageio.mimsave(os.path.join(gif_folder, 'seed-{}.gif'.format(SEED)), 
    #                             frames, duration=.04)
    
def save_data(steps,agent_x,agent_y,landmark_x,landmark_y,landmark_p_x,landmark_p_y,range_total,total_rewards, s_num = 0, d_num = 0):
    np.save(NAME_FOLDER+NAME_SUBFOLDER+'\\'+SCENARIOS[s_num]+'_'+DNNS[d_num]+'_steps.npy', steps)
    np.save(NAME_FOLDER+NAME_SUBFOLDER+'\\'+SCENARIOS[s_num]+'_'+DNNS[d_num]+'_agent_x.npy', agent_x)
    np.save(NAME_FOLDER+NAME_SUBFOLDER+'\\'+SCENARIOS[s_num]+'_'+DNNS[d_num]+'_agent_y.npy', agent_y)
    np.save(NAME_FOLDER+NAME_SUBFOLDER+'\\'+SCENARIOS[s_num]+'_'+DNNS[d_num]+'_landmark_x.npy', landmark_x)
    np.save(NAME_FOLDER+NAME_SUBFOLDER+'\\'+SCENARIOS[s_num]+'_'+DNNS[d_num]+'_landmark_y.npy', landmark_y)
    np.save(NAME_FOLDER+NAME_SUBFOLDER+'\\'+SCENARIOS[s_num]+'_'+DNNS[d_num]+'_landmark_p_x.npy', landmark_p_x)
    np.save(NAME_FOLDER+NAME_SUBFOLDER+'\\'+SCENARIOS[s_num]+'_'+DNNS[d_num]+'_landmark_p_y.npy', landmark_p_y)
    np.save(NAME_FOLDER+NAME_SUBFOLDER+'\\'+SCENARIOS[s_num]+'_'+DNNS[d_num]+'_range_total.npy', range_total)
    np.save(NAME_FOLDER+NAME_SUBFOLDER+'\\'+SCENARIOS[s_num]+'_'+DNNS[d_num]+'_total_rewards.npy', total_rewards)
    return
#%%
 
def colors(n): 
  ret = [] 
  r = int(random.random() * 256) 
  g = int(random.random() * 256) 
  b = int(random.random() * 256) 
  step = 256 / n 
  for i in range(n): 
    r += step 
    g += step 
    b += step 
    r = int(r) % 256 
    g = int(g) % 256 
    b = int(b) % 256 
    ret.append((r/256,g/256,b/256))  
  return ret    

if __name__=='__main__':
    steps,agent_x,agent_y,landmark_x,landmark_y,landmark_p_x,landmark_p_y,range_total,total_rewards = main()
    # save_data(steps,agent_x,agent_y,landmark_x,landmark_y,landmark_p_x,landmark_p_y,range_total,total_rewards)
    plot_test(steps,agent_x,agent_y,landmark_x,landmark_y,landmark_p_x,landmark_p_y,range_total,total_rewards)
    