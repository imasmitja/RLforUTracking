# main function that sets up environments
# perform training loop
####################################################################################################

from utilities import envs
from utilities.buffer import ReplayBuffer, ReplayBuffer_SummTree
from algorithms.ddpg.maddpg import MADDPG
from algorithms.td3.matd3_bc import MATD3_BC
from algorithms.sac.masac import MASAC
from algorithms.hrsac.mahrsac import MAHRSAC
import torch
import numpy as np
from tensorboardX import SummaryWriter
import os
from utilities.utilities import transpose_list, transpose_to_tensor
import time
import copy
import random
import pickle
import sys
from configparser import ConfigParser
# for saving gif
import imageio
import glob

def seeding(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)#as reproducibility docs
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False# as reproducibility docs
    torch.backends.cudnn.deterministic = True# as reproducibility docs

def pre_process(entity, batchsize):
    processed_entity = []
    for j in range(3):
        list = []
        for i in range(batchsize):
            b = entity[i][j]
            list.append(b)
        c = torch.Tensor(list)
        processed_entity.append(c)
    return processed_entity

def main():    
    # Read config file argument if its necessary
    if( len( sys.argv ) > 1 ):
        configFile = sys.argv[1]
    else:
        configFile = 'test_configuration'
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
    landmark_vel     = config.getfloat('hyperparam','landmark_vel')
    movement         = config.get('hyperparam','movement')
    pf_method        = config.getboolean('hyperparam','pf_method')
    rew_err_th       = config.getfloat('hyperparam','rew_err_th')
    rew_dis_th       = config.getfloat('hyperparam','rew_dis_th')
    max_range        = config.getfloat('hyperparam','max_range')
    max_current_vel  = config.getfloat('hyperparam','max_current_vel')
    range_dropping  = config.getfloat('hyperparam','range_dropping')
    
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
    try:
        max_vel   = config.getfloat('hyperparam','max_vel')
        random_vel= config.getboolean('hyperparam','random_vel')
    except:
        print('no max_vel or random_vel found in config file')
        max_vel = 0.
        random_vel = False
    
    #Chose device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") #To run the pytorch tensors on cuda GPU
    #DEVICE = 'cpu'

##################################################################################################    
     
    common_folder = r"/logs/"+configFile
    log_path = os.path.dirname(os.getcwd())+common_folder+r"/log"
    model_dir= os.path.dirname(os.getcwd())+common_folder+r"/model_dir"
    
    os.makedirs(model_dir, exist_ok=True)
    if PRE_TRAINED:
        PRE_TRAINED_EP = max([int(aux.split('_')[-1][:-3]) for i, aux in enumerate(glob.glob(model_dir+r'/episode_last_*.pt'))])
    else:
        PRE_TRAINED_EP = 0
        
    #print hyperparameters    
    print('Hyperparameters:')
    print('BUFFER_SIZE          =  ',BUFFER_SIZE)
    print('BATCH_SIZE           =  ',BATCH_SIZE)
    print('GAMMA                =  ',GAMMA)
    print('TAU                  =  ',TAU)
    print('LR_ACTOR             =  ',LR_ACTOR)
    print('LR_CRITIC            =  ',LR_CRITIC)
    print('WEIGHT_DECAY         =  ',WEIGHT_DECAY)
    print('UPDATE_EVERY         =  ',UPDATE_EVERY)
    print('UPDATE_TIMES         =  ',UPDATE_TIMES)
    print('SEED                 =  ',SEED)
    print('BENCHMARK            =  ',BENCHMARK)
    print('EXP_REP_BUF          =  ',EXP_REP_BUF)
    print('PRE_TRAINED          =  ',PRE_TRAINED)
    print('PRE_TRAINED_EP       =  ',PRE_TRAINED_EP)
    print('SCENARIO             =  ',SCENARIO)
    print('RNN activated        =  ',RNN)
    print('HISTORY_LENGTH       =  ',HISTORY_LENGTH)
    print('RENDER               =  ',RENDER)
    print('PROGRESS_BAR         =  ',PROGRESS_BAR)
    print('DEVICE               =  ',DEVICE)
    print('parallel_envs        =  ',parallel_envs)
    print('num_agents           =  ',num_agents)
    print('num_landmarks        =  ',num_landmarks)
    print('landmark_depth       =  ',landmark_depth)
    print('landmark_velocity    =  ',landmark_vel)
    print('number_of_episodes   =  ',number_of_episodes)
    print('episode_length       =  ',episode_length)
    print('save_interval        =  ',save_interval)
    print('noise                =  ',noise)
    print('noise_reduction      =  ',noise_reduction)
    print('DNN architecture     =  ',DNN)
    print('Alpha temperature    =  ',ALPHA)
    print('DNN Layer 1 size     =  ',DIM_1)
    print('DNN Layer 2 size     =  ',DIM_2)
    print('Folder name          =  ',common_folder)
    print('Model directory      =  ', model_dir)
    print('TIMESTAMP            =  ',time.strftime("%m%d%y_%H%M%S"))
    
    #Start the each seed
    seeding(seed = SEED+PRE_TRAINED_EP)
    t = 0
    
    if BENCHMARK:
        benchmark_dir = os.getcwd()+common_folder+r"/benchmark_dir"
        os.makedirs(benchmark_dir, exist_ok=True) 
        
    # initialize environment
    print('Initialize the number of parallel envs in torch')
    torch.set_num_threads(parallel_envs)
    print('Initialize the environments')
    env = envs.make_parallel_env(parallel_envs, SCENARIO, seed = SEED+PRE_TRAINED_EP, num_agents=num_agents, num_landmarks=num_landmarks, landmark_depth=landmark_depth, landmark_movable=landmark_movable, landmark_vel=landmark_vel, max_vel=max_vel, random_vel=random_vel, movement=movement, pf_method=pf_method, rew_err_th=rew_err_th, rew_dis_th=rew_dis_th, max_range=max_range, max_current_vel=max_current_vel,range_dropping=range_dropping, benchmark = BENCHMARK)
    
    # initialize replay buffer
    if EXP_REP_BUF == False:
        buffer = ReplayBuffer(int(BUFFER_SIZE))
    else:
        buffer = ReplayBuffer_SummTree(int(BUFFER_SIZE), SEED+PRE_TRAINED_EP) #Experienced replay buffer
        priority = np.ones(num_agents) #initial experienced replay buffer priority
                
    # initialize policy and critic
    print('Initialize the Actor-Critic networks')
    if DNN == 'MADDPG':
            maddpg =   MADDPG(num_agents = num_agents, num_landmarks = num_landmarks, landmark_depth=landmark_depth, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn = RNN, dim_1=DIM_1, dim_2=DIM_2)
    elif DNN == 'MATD3':
            maddpg = MATD3_BC(num_agents = num_agents, num_landmarks = num_landmarks, landmark_depth=landmark_depth, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn = RNN, dim_1=DIM_1, dim_2=DIM_2)
    elif DNN == 'MASAC':
            maddpg =    MASAC(num_agents = num_agents, num_landmarks = num_landmarks, landmark_depth=landmark_depth, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn = RNN, alpha = ALPHA, automatic_entropy_tuning = AUTOMATIC_ENTROPY, dim_1=DIM_1, dim_2=DIM_2)
     elif DNN == 'MAHRSAC':
            maddpg =    MAHRSAC(num_agents = num_agents, num_landmarks = num_landmarks, landmark_depth=landmark_depth, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE, rnn = RNN, alpha = ALPHA, automatic_entropy_tuning = AUTOMATIC_ENTROPY, dim_1=DIM_1, dim_2=DIM_2)
    
    else:
        print('ERROR UNKNOWN DNN ARCHITECTURE')
    logger = SummaryWriter(log_dir=log_path)
    
    agents_reward = []
    for n in range(num_agents):
        agents_reward.append([])
    
    if BENCHMARK:
        # placeholder for benchmarking info
        landmark_error_episode = []
        for i in range(num_landmarks):
            landmark_error_episode.append([1]) #we initialize the error at 1
        agent_outofworld_episode = []
        agent_collision_episode = []
        landmark_collision_episode = []
        for i in range(num_agents):
            agent_outofworld_episode.append([0]) #we initialize at 0
            agent_collision_episode.append([0]) #we initialize at 0
            landmark_collision_episode.append([0]) #we initialize at 0
    
    if PRE_TRAINED == True:
        #Load the pretrained agent's weights
        trained_checkpoint = model_dir + r'/episode'
        aux = torch.load(trained_checkpoint + '_last.pt')
        if DNN == 'MASAC' or DNN == 'MAHRSAC':
            with open(trained_checkpoint +  '_target_entropy_last.file', "rb") as f:
                target_entropy_aux = pickle.load(f)
            with open(trained_checkpoint +  '_log_alpha_last.file', "rb") as f:
                log_alpha_aux = pickle.load(f)
            with open(trained_checkpoint + '_alpha_last.file', "rb") as f:
                alpha_aux = pickle.load(f)
        for i in range(num_agents):  
            # load the weights from file
            if DNN == 'MADDPG':
                maddpg.maddpg_agent[i].actor.load_state_dict(aux[i]['actor_params'])
                maddpg.maddpg_agent[i].critic.load_state_dict(aux[i]['critic_params'])
                maddpg.maddpg_agent[i].target_actor.load_state_dict(aux[i]['target_actor_params'])
                maddpg.maddpg_agent[i].target_critic.load_state_dict(aux[i]['target_critic_params'])
                maddpg.maddpg_agent[i].actor_optimizer.load_state_dict(aux[i]['actor_optim_params'])
                maddpg.maddpg_agent[i].critic_optimizer.load_state_dict(aux[i]['critic_optim_params'])
            elif DNN == 'MATD3':
                maddpg.matd3_bc_agent[i].actor.load_state_dict(aux[i]['actor_params'])
                maddpg.matd3_bc_agent[i].critic.load_state_dict(aux[i]['critic_params'])
                maddpg.matd3_bc_agent[i].target_actor.load_state_dict(aux[i]['target_actor_params'])
                maddpg.matd3_bc_agent[i].target_critic.load_state_dict(aux[i]['target_critic_params'])
                maddpg.matd3_bc_agent[i].actor_optimizer.load_state_dict(aux[i]['actor_optim_params'])
                maddpg.matd3_bc_agent[i].critic_optimizer.load_state_dict(aux[i]['critic_optim_params'])
            elif DNN == 'MASAC' or DNN == 'MAHRSAC':
                if AUTOMATIC_ENTROPY:
                    maddpg.masac_agent[i].actor.load_state_dict(aux[i]['actor_params'])
                    maddpg.masac_agent[i].critic.load_state_dict(aux[i]['critic_params'])
                    maddpg.masac_agent[i].target_critic.load_state_dict(aux[i]['target_critic_params'])
                    maddpg.masac_agent[i].actor_optimizer.load_state_dict(aux[i]['actor_optim_params'])
                    maddpg.masac_agent[i].critic_optimizer.load_state_dict(aux[i]['critic_optim_params'])
                    maddpg.masac_agent[i].alpha_optimizer.load_state_dict(aux[i]['alpha_optim_params'])
                    #load agents alpha parameters
                    maddpg.masac_agent[i].target_entropy = target_entropy_aux[i]
                    maddpg.masac_agent[i].log_alpha = log_alpha_aux[i]
                    maddpg.masac_agent[i].alpha = alpha_aux[i]
                else:
                    maddpg.masac_agent[i].actor.load_state_dict(aux[i]['actor_params'])
                    maddpg.masac_agent[i].critic.load_state_dict(aux[i]['critic_params'])
                    maddpg.masac_agent[i].target_critic.load_state_dict(aux[i]['target_critic_params'])
                    maddpg.masac_agent[i].actor_optimizer.load_state_dict(aux[i]['actor_optim_params'])
                    maddpg.masac_agent[i].critic_optimizer.load_state_dict(aux[i]['critic_optim_params'])
            else:
                break
        
        #reload the replay buffer
        # import pdb; pdb.set_trace()
        buffer.reload(trained_checkpoint + r'_last.file')
        print('next')
        #reload agents reward
        with open(trained_checkpoint + r'_reward_last.file', "rb") as f:
            agents_reward = pickle.load(f)
        #reload landmark error
        with open(trained_checkpoint + r'_lerror_last.file', "rb") as f:
            landmark_error_episode = pickle.load(f)
        #reload agent out of world
        with open(trained_checkpoint + r'_outworld_last.file', "rb") as f:
            agent_outofworld_episode = pickle.load(f)
        #reload agent out of world
        with open(trained_checkpoint + r'_agentcoll_last.file', "rb") as f:
            agent_collision_episode = pickle.load(f)
        #reload agent out of world
        with open(trained_checkpoint + r'_landcoll_last.file', "rb") as f:
            landmark_collision_episode = pickle.load(f)

        #update the batch_size
        print('batch_size_was=',BATCH_SIZE)
        BATCH_SIZE *= 2**int(PRE_TRAINED_EP/200000)
        if BATCH_SIZE > 2048:
            BATCH_SIZE = 2048
        print('batch_size_is_now=',BATCH_SIZE)
        
        
    print('Starting iterations... \r\n')
    #show progress bar
    if PROGRESS_BAR == True:
        import tqdm
        #initializing progress bar object
        timer_bar = tqdm.tqdm(range(number_of_episodes),desc='\r\n Episode',position=0)
        
    counter = 0
    avg_rewards_best = -1000.
    for episode in range(0, number_of_episodes, parallel_envs):
        
        if PRE_TRAINED == True:
            episode += PRE_TRAINED_EP
            if episode == PRE_TRAINED_EP:
                noise *= noise_reduction**(int(PRE_TRAINED_EP/parallel_envs))

        if PROGRESS_BAR == True:
            #timer.update(episode)
            timer_bar.update(parallel_envs)

        #Reset the environment
        all_obs = env.reset() #[parallel_env, num_agents, observation_state_size], ex: [8,1,6]
        #Reset the noise
        for i in range(num_agents):
            if DNN == 'MADDPG':
                maddpg.maddpg_agent[i].noise.reset()
            elif DNN == 'MATD3':
                maddpg.matd3_bc_agent[i].noise.reset()
            elif DNN == 'MASAC' or DNN == 'MAHRSAC':
                maddpg.masac_agent[i].noise.reset()
            else:
                break
        #Reset the rewards
        reward_this_episode = np.zeros((parallel_envs, num_agents))  
        #Reset landmark error benchmark
        landmark_error = []
        for i in range(num_landmarks):
            landmark_error.append([])
        
        # flip the first two indices
        obs_roll = np.rollaxis(all_obs,1) #[num_agents, parallel_env, observation_state_size]
        obs = transpose_list(obs_roll) #list of size parallel_env, where each list index is an array of size observation_state_size
        
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
        
        frames = []
        tmax = 0
        # next_history = copy.deepcopy(history)       
        his = []
        if RENDER == True:
            frames.append(env.render('rgb_array'))
    
        for episode_t in range(episode_length):           
            # get actions
            # explore = only explore for a certain number of episodes
            # action input needs to be transposed
            # actions = maddpg.act(transpose_to_tensor(obs), noise=noise) 
            his = []
            for i in range(num_agents):
                his.append(torch.cat((transpose_to_tensor(history)[i],transpose_to_tensor(history_a)[i]), dim=2))
                      
            if episode < START_STEPS:
                #Uniform random steps at the begining as suggested by https://spinningup.openai.com/en/latest/algorithms/ddpg.html
                # actions_array = np.random.uniform(-1,1,(1,parallel_envs,num_agents))
                actions_array = np.random.uniform(-1,1,(num_agents,parallel_envs,1))
            else:
                actions = maddpg.act(his,transpose_to_tensor(obs) , noise=noise) 
                actions_array = torch.stack(actions).detach().numpy()
                        
            # transpose the list of list
            # flip the first two indices
            # input to step requires the first index to correspond to number of parallel agents
            actions_for_env = np.rollaxis(actions_array,1)
            
            # environment step
            # step forward one frame
            # next_obs, next_obs_full, rewards, dones, info = env.step(actions_for_env)
            next_obs, rewards, dones, info = env.step(actions_for_env)
            
            # rewards_sum += np.mean(rewards)
            
            # collect experience
            # add data to buffer
            # transition = (obs, obs_full, actions_for_env, rewards, next_obs, next_obs_full, dones)
            # transition = (obs, actions_for_env, rewards, next_obs, dones)
            # transition = (history, actions_for_env, rewards, next_history, dones)
            transition = (history, history_a, obs, actions_for_env, rewards, next_obs, dones)
            
            if EXP_REP_BUF == False:
                buffer.push(transition)
            else:
                buffer.push(transition,priority)                
            reward_this_episode += rewards

            # Update history buffers
            if RNN:
                # Add obs to the history buffer
                for n in range(parallel_envs):
                    for m in range(num_agents):
                        aux = obs[n][m].reshape(1,obs_size)
                        history[n][m] = np.concatenate((history[n][m],aux),axis=0)
                        history[n][m] = np.delete(history[n][m],0,0)
                # Add actions to the history buffer
                history_a = np.concatenate((history_a,actions_for_env.reshape(parallel_envs,num_agents,1,1)),axis=2)
                history_a = np.delete(history_a,0,2)
                    
            # obs, obs_full = next_obs, next_obs_full
            obs = next_obs
            
            # increment global step counter
            t += parallel_envs
            
            # save gif frame
            if RENDER == True:
                frames.append(env.render('rgb_array'))
                tmax+=1
                
            # for benchmarking learned policies
            if BENCHMARK:
                error_mean = np.zeros(num_landmarks)
                for e, inf in enumerate(info):
                    for l in range(num_landmarks):
                        # import pdb; pdb.set_trace()
                        error_mean[l] = np.add(error_mean[l],(inf['n'][0][0][l]))
                error_mean /= parallel_envs
                for i in range(num_landmarks):
                    landmark_error[i].append(error_mean[i])
                # for e, inf in enumerate(info):
                #     for a in range(num_agents):
                #         agent_info[a] = np.add(agent_info[a],(inf['n'][a]))
            
            # finish the episode if done
            if dones.any():
                # print('Number of episodes = ', episode_t)
                break
            
        #Reduce the quantity of noise added to the action
        noise *= noise_reduction
                
        # update once after every episode_per_update 
        # if len(buffer) > BATCH_SIZE and episode % episode_per_update < parallel_envs:
        if len(buffer) > BATCH_SIZE and episode % UPDATE_EVERY < parallel_envs:
            for _ in range(UPDATE_TIMES):
                priority = np.zeros(num_agents)
                for a_i in range(num_agents):
                    if EXP_REP_BUF == False:
                        samples = buffer.sample(BATCH_SIZE)
                        priority = maddpg.update(samples, a_i, logger)
                    else:
                        samples, indexes = buffer.sample(BATCH_SIZE)
                        new_priorities = maddpg.update(samples, a_i, logger)
                        priority[a_i] = buffer.update(indexes, new_priorities)
                if EXP_REP_BUF == True:
                    priority /= num_agents
            maddpg.update_targets() #soft update the target network towards the actual networks

        for i in range(parallel_envs):
            for n in range(num_agents):
                agents_reward[n].append(reward_this_episode[i,n])
                if len(agents_reward[n]) > REWARD_WINDOWS:
                    agents_reward[n] = agents_reward[n][1:]
                    
        if BENCHMARK and episode_t>180:
            for i in range(num_landmarks):
                landmark_error_episode[i].append(np.array(landmark_error[i][-100:]).mean())
                if len(landmark_error_episode[i]) > LANDMARK_ERROR_WINDOWS:
                    landmark_error_episode[i] = landmark_error_episode[i][1:]
        
        if BENCHMARK:
            for ii in range(num_agents):
                agent_outofworld = 0
                landmark_collision = 0
                agent_collision = 0
                for i, inf in enumerate(info):
                    #info strucutre: (world.error,landmarks_real_p, self.agent_outofworld, self.landmark_collision, self.agent_collision)
                    agent_outofworld += inf['n'][ii][2]
                    landmark_collision += inf['n'][ii][3]
                    agent_collision += inf['n'][ii][4]
                #append it to the historical list
                agent_outofworld_episode[ii].append(agent_outofworld)
                landmark_collision_episode[ii].append(landmark_collision)
                agent_collision_episode[ii].append(agent_collision)
                if len(agent_outofworld_episode[ii]) > COLLISION_OUTWORLD_WINDOWS:
                    agent_outofworld_episode[ii] = agent_outofworld_episode[ii][1:]
                    landmark_collision_episode[ii] = landmark_collision_episode[ii][1:]
                    agent_collision_episode[ii] = agent_collision_episode[ii][1:]
        if episode % 1000 < parallel_envs or episode == number_of_episodes-1:
            if (PRE_TRAINED == True and episode == PRE_TRAINED_EP):
                #Don't save the first iteration of a pretrined network
                pass
            else:
                avg_rewards = []
                std_rewards = []
                for n in range(num_agents):
                    avg_rewards.append(np.mean(agents_reward[n]))
                    std_rewards.append(np.std(agents_reward[n])) 
                for a_i, avg_rew in enumerate(avg_rewards):
                    logger.add_scalar('agent%i/mean_episode_rewards' % a_i, avg_rew, episode)
                    logger.add_scalar('agent%i/std_episode_rewards' % a_i, std_rewards[a_i], episode)
                    if BENCHMARK:
                        logger.add_scalar('agent%i/agent_outofworld_episode' % a_i, np.array(agent_outofworld_episode[a_i]).sum(), episode)
                        logger.add_scalar('agent%i/landmark_collision_episode' % a_i, np.array(landmark_collision_episode[a_i]).sum(), episode)
                        logger.add_scalar('agent%i/agent_collision_episode' % a_i, np.array(agent_collision_episode[a_i]).sum(), episode)
                if BENCHMARK:
                    for l_i, err in enumerate(landmark_error_episode):
                        # import pdb; pdb.set_trace()
                        logger.add_scalar('landmark%i/mean_episode_error' % l_i, np.array(err).mean(), episode)
                        logger.add_scalar('landmark%i/std_episode_error' % l_i, np.array(err).std(), episode)
                    if PROGRESS_BAR == True:
                        timer_bar.set_postfix({'avg_rew': avg_rew, 'avg_error': np.array(err).mean()})
                else:
                    if PROGRESS_BAR == True:
                        timer_bar.set_postfix({'avg_rew': avg_rew})
                         
        if counter > 400000:
            #increase batch_size as:https://arxiv.org/pdf/1711.00489.pdf
            print('batch_size_was=',BATCH_SIZE)
            BATCH_SIZE *= 2
            if BATCH_SIZE > 2048:
                BATCH_SIZE = 2048
            print('batch_size_is_now=',BATCH_SIZE)
            counter = 0
        counter += parallel_envs
            
        #saving model
        # save info or not
        if PRE_TRAINED == True:
            aux_episode = PRE_TRAINED_EP + 0
        else:
            aux_episode = 0
        save_info = (((episode) % save_interval < parallel_envs and episode > aux_episode) or episode==number_of_episodes-parallel_envs)
        save_dict_list =[]
        target_entropy_list = []
        log_alpha_list = []
        alpha_list = []
        if save_info:
            for i in range(num_agents):
                
                if DNN == 'MADDPG':
                    save_dict = {'actor_params' : maddpg.maddpg_agent[i].actor.state_dict(),
                             'target_actor_params' : maddpg.maddpg_agent[i].target_actor.state_dict(),
                             'actor_optim_params': maddpg.maddpg_agent[i].actor_optimizer.state_dict(),
                             'critic_params' : maddpg.maddpg_agent[i].critic.state_dict(),
                             'target_critic_params' : maddpg.maddpg_agent[i].target_critic.state_dict(),
                             'critic_optim_params' : maddpg.maddpg_agent[i].critic_optimizer.state_dict()}
                elif DNN == 'MATD3':
                    save_dict = {'actor_params' : maddpg.matd3_bc_agent[i].actor.state_dict(),
                             'target_actor_params' : maddpg.matd3_bc_agent[i].target_actor.state_dict(),
                             'actor_optim_params': maddpg.matd3_bc_agent[i].actor_optimizer.state_dict(),
                             'critic_params' : maddpg.matd3_bc_agent[i].critic.state_dict(),
                             'target_critic_params' : maddpg.matd3_bc_agent[i].target_critic.state_dict(),
                             'critic_optim_params' : maddpg.matd3_bc_agent[i].critic_optimizer.state_dict()}
                elif DNN == 'MASAC' or DNN == 'MAHRSAC':
                    if AUTOMATIC_ENTROPY:
                        save_dict = {'actor_params' : maddpg.masac_agent[i].actor.state_dict(),
                                 'actor_optim_params': maddpg.masac_agent[i].actor_optimizer.state_dict(),
                                 'critic_params' : maddpg.masac_agent[i].critic.state_dict(),
                                 'target_critic_params' : maddpg.masac_agent[i].target_critic.state_dict(),
                                 'critic_optim_params' : maddpg.masac_agent[i].critic_optimizer.state_dict(),
                                 'alpha_optim_params' : maddpg.masac_agent[i].alpha_optimizer.state_dict()}
                        #Append agents alpha parameters
                        target_entropy_list.append(maddpg.masac_agent[i].target_entropy)
                        log_alpha_list.append(maddpg.masac_agent[i].log_alpha)
                        alpha_list.append(maddpg.masac_agent[i].alpha)
                    else:
                        save_dict = {'actor_params' : maddpg.masac_agent[i].actor.state_dict(),
                                 'actor_optim_params': maddpg.masac_agent[i].actor_optimizer.state_dict(),
                                 'critic_params' : maddpg.masac_agent[i].critic.state_dict(),
                                 'target_critic_params' : maddpg.masac_agent[i].target_critic.state_dict(),
                                 'critic_optim_params' : maddpg.masac_agent[i].critic_optimizer.state_dict()}
                        
                else:
                    break
            
                save_dict_list.append(save_dict)
                
            #SAVE LAST VALUES
            #save num episode
            torch.save([], 
                       os.path.join(model_dir, 'episode_last_{}.pt'.format(episode)))
            #save dict_list
            torch.save(save_dict_list, 
                       os.path.join(model_dir, 'episode_last.pt'))
            #save the replay buffer
            buffer.save(os.path.join(model_dir, 'episode_last.file'))
            #save agents reward
            with open(os.path.join(model_dir, 'episode_reward_last.file'), "wb") as f:
                pickle.dump(agents_reward, f)
            #save landmark error
            with open(os.path.join(model_dir, 'episode_lerror_last.file'), "wb") as f:
                pickle.dump(landmark_error_episode, f)
            #reload agent out of world
            with open(os.path.join(model_dir, 'episode_outworld_last.file'), "wb") as f:
                pickle.dump(agent_outofworld_episode, f)
            #reload agent collisions
            with open(os.path.join(model_dir, 'episode_agentcoll_last.file'), "wb") as f:
                pickle.dump(agent_collision_episode, f)
            #reload landmark collisions
            with open(os.path.join(model_dir, 'episode_landcoll_last.file'), "wb") as f:
                pickle.dump(landmark_collision_episode, f)
            #save agents alpha parameters
            with open(os.path.join(model_dir, 'episode_target_entropy_last.file'), "wb") as f:
                pickle.dump(target_entropy_list, f)
            with open(os.path.join(model_dir, 'episode_log_alpha_last.file'), "wb") as f:
                pickle.dump(log_alpha_list, f)
            with open(os.path.join(model_dir, 'episode_alpha_last.file'), "wb") as f:
                pickle.dump(alpha_list, f)
                
            if np.mean(avg_rewards) > np.mean(avg_rewards_best):
                #SAVE BEST VALUES
                #save num episode
                torch.save([], 
                           os.path.join(model_dir, 'episode_best_{}.pt'.format(episode)))
                #save dict_list
                torch.save(save_dict_list, 
                           os.path.join(model_dir, 'episode_best.pt'))
                #save the replay buffer
                buffer.save(os.path.join(model_dir, 'episode_best.file'))
                #save agents reward
                with open(os.path.join(model_dir, 'episode_reward_best.file'), "wb") as f:
                    pickle.dump(agents_reward, f)
                #save landmark error
                with open(os.path.join(model_dir, 'episode_lerror_best.file'), "wb") as f:
                    pickle.dump(landmark_error_episode, f)
                #reload agent out of world
                with open(os.path.join(model_dir, 'episode_outworld_best.file'), "wb") as f:
                    pickle.dump(agent_outofworld_episode, f)
                #reload agent collisions
                with open(os.path.join(model_dir, 'episode_agentcoll_best.file'), "wb") as f:
                    pickle.dump(agent_collision_episode, f)
                #reload landmark collisions
                with open(os.path.join(model_dir, 'episode_landcoll_best.file'), "wb") as f:
                    pickle.dump(landmark_collision_episode, f)
                #save agents alpha parameters
                with open(os.path.join(model_dir, 'episode_target_entropy_best.file'), "wb") as f:
                    pickle.dump(target_entropy_list, f)
                with open(os.path.join(model_dir, 'episode_log_alpha_best.file'), "wb") as f:
                    pickle.dump(log_alpha_list, f)
                with open(os.path.join(model_dir, 'episode_alpha_best.file'), "wb") as f:
                    pickle.dump(alpha_list, f)
                #update avg_rewards_best
                try:
                    avg_rewards_best = avg_rewards.copy()
                except:
                    pass
                
            if RENDER == True:
                # save gif files
                imageio.mimsave(os.path.join(model_dir, 'episode-{}.gif'.format(episode)), 
                                frames, duration=.04)
            
            #save benchmark
            # if BENCHMARK:
            #     file1 = open(benchmark_dir+r"\episode-{}.txt".format(episode),"w")#append mode 
            #     file1.write(str(np.array(agent_info)/t)) 
            #     file1.close() 

    env.close()
    logger.close()
    #timer.finish()

if __name__=='__main__':
    print('Start main')
    main()
    
    

