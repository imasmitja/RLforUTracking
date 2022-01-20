# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 20:37:13 2021

@author: Usuari

5. Watch a Smart Agent!
In the next code cell, you will load the trained weights from file to watch a smart agent!
"""
from utilities import envs
from utilities.buffer import ReplayBuffer, ReplayBuffer_SummTree
from algorithms.ddpg.maddpg import MADDPG
from algorithms.sac.masac import MASAC
from algorithms.td3.matd3_bc import MATD3_BC
import torch
import numpy as np
from tensorboardX import SummaryWriter
import os
from utilities.utilities import transpose_list, transpose_to_tensor, circle_path, random_levy
import time
import copy
import matplotlib.pyplot as plt
import pickle
import sys
from configparser import ConfigParser

# for saving gif
import imageio

# Read config file argument if its necessary
if( len( sys.argv ) > 1 ):
    configFile = sys.argv[1]
else:
    configFile = 'trained_saca'
print ('Configuration File   =  ',configFile +'.txt')

config = ConfigParser()
config.read(configFile+'.txt')

BUFFER_SIZE    = config.getint('hyperparam','BUFFER_SIZE')
BATCH_SIZE     = config.getint('hyperparam','BATCH_SIZE')
GAMMA          = config.getfloat('hyperparam','GAMMA')
TAU            = config.getfloat('hyperparam','TAU')
LR_ACTOR       = config.getfloat('hyperparam','LR_ACTOR')
LR_CRITIC      = config.getfloat('hyperparam','LR_CRITIC')
WEIGHT_DECAY   = config.getfloat('hyperparam','WEIGHT_DECAY')
UPDATE_EVERY   = config.getint('hyperparam','UPDATE_EVERY')
UPDATE_TIMES   = config.getint('hyperparam','UPDATE_TIMES')
SEED           = config.getint('hyperparam','SEED')
BENCHMARK      = config.getboolean('hyperparam','BENCHMARK')
EXP_REP_BUF    = config.getboolean('hyperparam','EXP_REP_BUF')
PRE_TRAINED    = config.getboolean('hyperparam','PRE_TRAINED')
#Scenario used to train the networks
SCENARIO       = config.get('hyperparam','SCENARIO')
RENDER         = config.getboolean('hyperparam','RENDER')
PROGRESS_BAR   = config.getboolean('hyperparam','PROGRESS_BAR')
RNN            = config.getboolean('hyperparam','RNN')
HISTORY_LENGTH = config.getint('hyperparam','HISTORY_LENGTH')
DNN            = config.get('hyperparam','DNN')
START_STEPS    = config.getint('hyperparam','START_STEPS')
REWARD_WINDOWS = config.getint('hyperparam','REWARD_WINDOWS')
LANDMARK_ERROR_WINDOWS     = config.getint('hyperparam','LANDMARK_ERROR_WINDOWS')
COLLISION_OUTWORLD_WINDOWS = config.getint('hyperparam','COLLISION_OUTWORLD_WINDOWS')
ALPHA          = config.getfloat('hyperparam','ALPHA')
AUTOMATIC_ENTROPY = config.getboolean('hyperparam','AUTOMATIC_ENTROPY')
DIM_1          = config.getint('hyperparam','DIM_1')
DIM_2          = config.getint('hyperparam','DIM_2')
# number of parallel agents
parallel_envs  = config.getint('hyperparam','parallel_envs')
# number of agents per environment
num_agents     = config.getint('hyperparam','num_agents')
# number of landmarks (or targets) per environment
num_landmarks  = config.getint('hyperparam','num_landmarks')
landmark_depth = config.getfloat('hyperparam','landmark_depth')
landmark_movable = config.getboolean('hyperparam','landmark_movable')
movement         = config.get('hyperparam','movement')
pf_method        = config.getboolean('hyperparam','pf_method')
rew_err_th       = config.getfloat('hyperparam','rew_err_th')
rew_dis_th       = config.getfloat('hyperparam','rew_dis_th')
# number of training episodes.
# change this to higher number to experiment. say 30000.
number_of_episodes = config.getint('hyperparam','number_of_episodes')
episode_length = config.getint('hyperparam','episode_length')
# how many episodes to save policy and gif
save_interval  = config.getint('hyperparam','save_interval')
# amplitude of OU noise
# this slowly decreases to 0
noise          = config.getfloat('hyperparam','noise')
noise_reduction= config.getfloat('hyperparam','noise_reduction')    
fol_in = int(np.random.rand()*1000)

#Chose device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") #To run the pytorch tensors on cuda GPU
#DEVICE = 'cpu'

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
        
        #New test using LSTM
        trained_checkpoint = os.getcwd()+'/logs/' + configFile+ '/model_dir/episode' #Test SAC auto
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
    