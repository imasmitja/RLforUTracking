# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from algorithms.td3.td3_bc import TD3_BCAgent
import torch
from utilities.utilities import soft_update, transpose_to_tensor, transpose_list, gumbel_softmax
import numpy as np



class MATD3_BC:
    def __init__(self, num_agents = 3, num_landmarks = 1, landmark_depth=15., discount_factor=0.95, tau=0.02, lr_actor=1.0e-2, lr_critic=1.0e-2, weight_decay=1.0e-5, device = 'cpu', rnn = True, dim_1=64, dim_2=32):
        super(MATD3_BC, self).__init__()

        # ([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + [entity_range] + [entity_depth] + [agent.state.p_pos_origin]) + action(for critic not actor)
        in_actor = 1*2*2 + num_landmarks*2 + (num_agents-1)*2 + num_landmarks + 1*num_landmarks + 2 +1#test with target depth and agent's origin for science

        hidden_in_actor = dim_2
        hidden_out_actor = int(hidden_in_actor/2)
        out_actor = 1 #each agent have 2 continuous actions on x-y plane
        in_critic = in_actor * num_agents # the critic input is all agents concatenated
        hidden_in_critic = dim_2
        hidden_out_critic = int(hidden_in_critic/2)
        #RNN
        rnn_num_layers = 2 #two stacked RNN to improve the performance (default = 1)
        rnn_hidden_size_actor = dim_1
        rnn_hidden_size_critic = dim_1
        
        # print('Actor NN configuration:')
        # print('Input nodes number:            ',in_actor)
        # print('Hidden 1st layer nodes number: ',hidden_in_actor)
        # print('Hidden 2nd layer nodes number: ',hidden_out_actor)
        # print('Output nodes number:           ',out_actor)
        # print('RNN hidden size actor :        ',rnn_hidden_size_actor)
        # print('Critic NN configuration:')
        # print('Input nodes number:            ',in_critic)
        # print('Hidden 1st layer nodes number: ',hidden_in_critic)
        # print('Hidden 2nd layer nodes number: ',hidden_out_critic)
        # print('Output nodes number:           ',out_actor)
        # print('RNN hidden size critic:        ',rnn_hidden_size_critic)
        
        self.matd3_bc_agent = [TD3_BCAgent(in_actor, hidden_in_actor, hidden_out_actor, out_actor, in_critic, hidden_in_critic, hidden_out_critic, rnn_num_layers, rnn_hidden_size_actor, rnn_hidden_size_critic, lr_actor=lr_actor, lr_critic=lr_critic, weight_decay=weight_decay, device=device, rnn = rnn) for _ in range(num_agents)]
        # self.matd3_bc_agent = [DDPGAgent(14, 128, 128, 2, 48, 128, 128, lr_actor=lr_actor, lr_critic=lr_critic, weight_decay=weight_decay, device=device) for _ in range(num_agents)]
        
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0
        self.iter_delay = 0
        
        self.policy_freq = 2
        self.num_agents = num_agents
        
        #initial priority for the experienced replay buffer
        self.priority = 1.
        
        
        #device 'cuda' or 'cpu'
        self.device = device

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [td3_bc_agent.actor for td3_bc_agent in self.matd3_bc_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [td3_bc_agent.target_actor for td3_bc_agent in self.matd3_bc_agent]
        return target_actors

    def act(self, his_all_agents, obs_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        actions_next = [agent.act(his, obs, noise) for agent, his, obs in zip(self.matd3_bc_agent, his_all_agents, obs_all_agents)]
        return actions_next

    def target_act(self, his_all_agents, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        target_actions_next = [td3_bc_agent.target_act(his, obs, noise) for td3_bc_agent, his, obs in zip(self.matd3_bc_agent, his_all_agents, obs_all_agents)]
        return target_actions_next

    def update(self, samples, agent_number, logger):
        """update the critics and actors of all the agents 
            Update parameters of agent model based on sample from replay buffer
            Inputs:
                samples: tuple of (observations, full observations, actions, rewards, next
                        observations, full next observations, and episode end masks) sampled randomly from
                        the replay buffer. Each is a list with entries
                        corresponding to each agent
                agent_number (int): index of agent to update
                logger (SummaryWriter from Tensorboard-Pytorch):
                    If passed in, important quantities will be logged
        """
        # need to transpose each element of the samples
        # to flip obs[parallel_agent][agent_number] to
        # obs[agent_number][parallel_agent]
        # obs, obs_full, action, reward, next_obs, next_obs_full, done = map(transpose_to_tensor, samples)
        his_obs, his_act, obs, action, reward, next_obs, done = map(transpose_to_tensor, samples)
        
        # obs_full = torch.stack(obs_full)
        # next_obs_full = torch.stack(next_obs_full)
        
        obs_full = torch.cat(obs, dim=1)
        next_obs_full = torch.cat(next_obs, dim=1)
        action = torch.cat(action, dim=1)
        obs_act_full = torch.cat((obs_full,action), dim=1)
        his = []
        for i in range(len(his_obs)):
            his.append(torch.cat((his_obs[i],his_act[i]), dim=2))
        his_full = torch.cat(his,dim=2)
        
        # next_his = []
        # for i in range(len(his_obs)):
        #     aux = torch.cat((his_obs[i],obs[i].reshape(his_obs[i].shape[0],1,his_obs[i].shape[2])),dim=1)
        #     aux = np.delete(aux,0,1)
        #     aux_a = torch.cat((his_act[i],action.reshape(his_act[i].shape[0],1,his_act[i].shape[2])),dim=1)
        #     aux_a = np.delete(aux_a,0,1)
        #     next_his.append(torch.cat((aux,aux_a), dim=2))

        
        agent = self.matd3_bc_agent[agent_number]
        agent.critic_optimizer.zero_grad()

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        #critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(st+1,at+1) from target network
        
        target_actions_next = self.target_act(his,next_obs) 
        # target_actions_next = self.target_act(next_his,next_obs) 
        
        target_actions_next = torch.cat(target_actions_next, dim=1)
        # target_critic_input = torch.cat((next_obs_full.t(),target_actions_next), dim=1).to(self.device)
        # target_critic_input = torch.cat((next_obs_full,target_actions_next), dim=1).to(self.device)
        next_his_full = torch.cat((his_full[:,1:,:],obs_act_full.reshape(obs_act_full.shape[0],1,obs_act_full.shape[1])),dim=1)
        next_obs_act_full = torch.cat((next_obs_full,target_actions_next), dim=1)
        
        # Compute Q targets (y) for current states (y_i)
        with torch.no_grad():
            target_Q1, target_Q2 = agent.target_critic(next_his_full.to(self.device), next_obs_act_full.to(self.device))
            target_Q = torch.min(target_Q1, target_Q2)
            y = reward[agent_number].view(-1, 1).to(self.device) + self.discount_factor * target_Q * (1 - done[agent_number].view(-1, 1)).to(self.device)

        # Compute Q expected (q) 
        # critic_input = torch.cat((obs_full.t(), action), dim=1).to(self.device)
        # critic_input = torch.cat((obs_full, action), dim=1).to(self.device)
        current_Q1, current_Q2 = agent.critic(his_full.to(self.device), obs_act_full.to(self.device))
        
        # Priorized Experience Replay
        # aux = abs(q - y.detach()) + 0.1 #we introduce a fixed small constant number to avoid priorities = 0.
        # aux = np.matrix(aux.detach().numpy())
        # new_priorities = np.sqrt(np.diag(aux*aux.T))
        
        # import pdb; pdb.set_trace()
        # Compute critic loss
        # huber_loss = torch.nn.SmoothL1Loss()
        # critic_loss = huber_loss(q, y.detach())
        # Compute critic loss
        loss_mse = torch.nn.MSELoss()
        critic_loss = loss_mse(current_Q1, y.detach()) + loss_mse(current_Q2, y.detach())
        
        # Minimize the loss
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.critic_optimizer.step()
        
        # Delayed policy updates
        if self.iter_delay % self.policy_freq == 0:
            # ---------------------------- update actor ---------------------------- #
            #update actor network using policy gradient
            # Compute actor loss
            agent.actor_optimizer.zero_grad()
            # make input to agent
            curr_q_input = self.matd3_bc_agent[agent_number].actor(his[agent_number].to(self.device), obs[agent_number].to(self.device))
            # use Gumbel-Softmax sample
            # curr_q_input = gumbel_softmax(curr_q_input, hard = True) # this should be used only if the action is discrete (for example in comunications, but in general the action is not discrete)
            # detach the other agents to save computation
            # saves some time for computing derivative
            # q_input = [ self.matd3_bc_agent[i].actor(ob.to(self.device)) if i == agent_number \
            #            else self.matd3_bc_agent[i].actor(ob.to(self.device)).detach()
            #            for i, ob in enumerate(obs) ]
            q_input = [ curr_q_input if i == agent_number \
                       else self.matd3_bc_agent[i].actor(his[i].to(self.device),ob.to(self.device)).detach()
                       for i, ob in enumerate(obs) ]
                    
            q_input = torch.cat(q_input, dim=1)
            # combine all the actions and observations for input to critic
            # many of the obs are redundant, and obs[1] contains all useful information already
            # q_input2 = torch.cat((obs_full.t(), q_input), dim=1)
            # q_input2 = torch.cat((obs_full.to(self.device), q_input), dim=1)
            obs_q_full = torch.cat((obs_full.to(self.device),q_input), dim=1)
            actor_loss = -agent.critic.Q1(his_full.to(self.device),obs_q_full).mean() # get the policy gradient
            actor_loss += (curr_q_input**2).mean()*1e-3 #modification from https://github.com/shariqiqbal2810/maddpg-pytorch/blob/master/algorithms/maddpg.py
            
            # Minimize the loss
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
            agent.actor_optimizer.step()
    
            al = actor_loss.cpu().detach().item()
            cl = critic_loss.cpu().detach().item()
            logger.add_scalars('agent%i/losses' % agent_number,
                               {'critic loss': cl,
                                'actor_loss': al},
                               self.iter)
        
        
        # if agent_number+1 == self.num_agents: #this works test 78
        #     self.iter_delay += 1

    def update_targets(self):
        """soft update targets"""
        self.iter += 1 #this doesnt work as well as the other test 80
        self.iter_delay += 1
        # ----------------------- update target networks ----------------------- #
        for td3_bc_agent in self.matd3_bc_agent:
            soft_update(td3_bc_agent.target_actor, td3_bc_agent.actor, self.tau)
            soft_update(td3_bc_agent.target_critic, td3_bc_agent.critic, self.tau)
            
            
            




