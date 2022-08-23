# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from algorithms.hrsac.hrsac import HRSACAgent
import torch
from utilities.utilities import soft_update, transpose_to_tensor, transpose_list, gumbel_softmax
import numpy as np

'''
An addaption from:

Code partially extracted from:
https://github.com/denisyarats/pytorch_sac/blob/81c5b536d3a1c5616b2531e446450df412a064fb/agent/sac.py
https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/SAC/sac_torch.py
https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/sac.py

'''

class MAHRSAC:
    def __init__(self, num_agents = 3, num_landmarks = 1, landmark_depth=15., discount_factor=0.95, tau=0.02, lr_actor=1.0e-2, lr_critic=1.0e-2, weight_decay=1.0e-5, device = 'cpu', rnn = True, alpha = 0.2, automatic_entropy_tuning = True, dim_1=64, dim_2=32):
        super(MAHRSAC, self).__init__()

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
        
        self.masac_agent = [HRSACAgent(in_actor, hidden_in_actor, hidden_out_actor, out_actor, in_critic, hidden_in_critic, hidden_out_critic, rnn_num_layers, rnn_hidden_size_actor, rnn_hidden_size_critic, lr_actor=lr_actor, lr_critic=lr_critic, weight_decay=weight_decay, device=device, rnn = rnn, alpha = alpha, automatic_entropy_tuning=automatic_entropy_tuning) for _ in range(num_agents)]
        
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
        
        #To update alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [sac_agent.actor for sac_agent in self.masac_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [sac_agent.target_actor for sac_agent in self.masac_agent]
        return target_actors

    def act(self, his_all_agents, obs_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        actions_next = [agent.act(his, obs, noise) for agent, his, obs in zip(self.masac_agent, his_all_agents, obs_all_agents)]
        return actions_next

    def act_prob(self, his_all_agents, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        actions_next = []
        log_probs = []
        for sac_agent, his, obs in zip(self.masac_agent, his_all_agents, obs_all_agents):
            action, log_prob = sac_agent.act_prob(his, obs, noise)
            log_prob = log_prob.view(-1)
            actions_next.append(action)
            log_probs.append(log_prob)
        # target_actions_next = [sac_agent.target_actor.sample_normal(his, obs, noise) for sac_agent, his, obs in zip(self.masac_agent, his_all_agents, obs_all_agents)]
        # for i,aux in enumerate(log_probs):
        #     log_probs[i]=aux.view(-1,1)
        return actions_next, log_probs

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

        # import pdb; pdb.set_trace()
        
        agent = self.masac_agent[agent_number]
        agent.critic_optimizer.zero_grad()

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        #critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(st+1,at+1) from target network
        
        # target_actions_next = self.target_act(his,next_obs) 
        # target_actions_next, log_probs = self.target_act(his, next_obs)
        actions_next, log_probs = self.act_prob(his, next_obs, noise=0.) #Unlike in TD3, the next-state actions used in the target come from the current policy instead of a target policy.
        actions_next = torch.cat(actions_next, dim=1)
        # log_probs = torch.cat(log_probs, dim=0)
        next_his_full = torch.cat((his_full[:,1:,:],obs_act_full.reshape(obs_act_full.shape[0],1,obs_act_full.shape[1])),dim=1)
        next_obs_act_full = torch.cat((next_obs_full,actions_next), dim=1)
        
        # Compute Q targets (y) for current states (y_i)
        with torch.no_grad():
            target_Q1, target_Q2 = agent.target_critic(next_his_full.to(self.device), next_obs_act_full.to(self.device))
            target_V = torch.min(target_Q1, target_Q2) - agent.alpha*log_probs[agent_number].view(-1,1)
            target_Q = reward[agent_number].view(-1, 1).to(self.device) + self.discount_factor * target_V * (1 - done[agent_number].view(-1, 1)).to(self.device)

        # Compute Q expected (q) 
        current_Q1, current_Q2 = agent.critic(his_full.to(self.device), obs_act_full.to(self.device))
        
        # Compute critic loss
        loss_mse = torch.nn.MSELoss()
        critic_loss = loss_mse(current_Q1, target_Q.detach()) + loss_mse(current_Q2, target_Q.detach())
        
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
            # curr_q_input = self.masac_agent[agent_number].actor(his[agent_number].to(self.device), obs[agent_number].to(self.device))
            # actions, log_probs = self.masac_agent[agent_number].actor.sample_normal(his[agent_number].to(self.device), obs[agent_number].to(self.device))
            actions, log_probs = self.masac_agent[agent_number].act_prob(his[agent_number].to(self.device), obs[agent_number].to(self.device),noise=1.)
            log_probs = log_probs.view(-1)
            # use Gumbel-Softmax sample
            # curr_q_input = gumbel_softmax(curr_q_input, hard = True) # this should be used only if the action is discrete (for example in comunications, but in general the action is not discrete)
            # detach the other agents to save computation
            # saves some time for computing derivative
            # q_input = [ self.masac_agent[i].actor(ob.to(self.device)) if i == agent_number \
            #            else self.masac_agent[i].actor(ob.to(self.device)).detach()
            #            for i, ob in enumerate(obs) ]
            # q_input = [ curr_q_input if i == agent_number \
            #            else self.masac_agent[i].actor.sample_normal(his[i].to(self.device),ob.to(self.device)).detach()
            #            for i, ob in enumerate(obs) ]
                
            q_actions = []
            q_log_probs = []
            for i, ob in enumerate(obs):
                if i == agent_number:
                    q_actions.append(actions)
                    q_log_probs.append(log_probs)
                else:
                    actions_aux, log_probs_aux = self.masac_agent[i].actor.sample_normal(his[i].to(self.device),ob.to(self.device))
                    log_probs_aux = log_probs_aux.view(-1)
                    q_actions.append(actions_aux.detach())
                    q_log_probs.append(log_probs_aux.detach())   
              
            q_actions = torch.cat(q_actions, dim=1)
            # q_log_probs = torch.cat(q_log_probs, dim=0)
            # combine all the actions and observations for input to critic
            # many of the obs are redundant, and obs[1] contains all useful information already
            obs_q_full = torch.cat((obs_full.to(self.device),q_actions.to(self.device)), dim=1)
            actor_Q1, actor_Q2 = agent.critic(his_full.to(self.device),obs_q_full)
            actor_Q = torch.min(actor_Q1, actor_Q2)
            actor_loss = (agent.alpha*q_log_probs[agent_number].view(-1,1)-actor_Q).mean() # get the policy gradient
            
            # Minimize the loss
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
            agent.actor_optimizer.step()
            
            #Update alpha
            if self.automatic_entropy_tuning:
                # import pdb; pdb.set_trace()
                alpha_loss = -(agent.log_alpha * (q_log_probs[agent_number].view(-1,1) + agent.target_entropy).detach()).mean()    
                agent.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                agent.alpha_optimizer.step()    
                agent.alpha = agent.log_alpha.exp()
    
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
        for sac_agent in self.masac_agent:
            # soft_update(sac_agent.target_actor, sac_agent.actor, self.tau)
            soft_update(sac_agent.target_critic, sac_agent.critic, self.tau)
            
            
            
            




