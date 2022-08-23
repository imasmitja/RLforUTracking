# individual network settings for each actor + critic pair
# see networkforall for details
'''
An addaption from:

Code partially extracted from:
https://github.com/denisyarats/pytorch_sac/blob/81c5b536d3a1c5616b2531e446450df412a064fb/agent/sac.py
https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/SAC/sac_torch.py
https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/sac.py


'''

from algorithms.sac.networkforall_sac import Network
from utilities.utilities import hard_update
from torch.optim import Adam
import torch
import numpy as np


# add OU noise for exploration
from utilities.OUNoise import OUNoise

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

class SACAgent():
    def __init__(self, in_actor, hidden_in_actor, hidden_out_actor, out_actor, in_critic, hidden_in_critic, hidden_out_critic, rnn_num_layers, rnn_hidden_size_actor, rnn_hidden_size_critic , lr_actor=1.0e-2, lr_critic=1.0e-2, weight_decay=1.0e-5, device = 'cpu', rnn = True, alpha = 0.2, automatic_entropy_tuning = True):
        super(SACAgent, self).__init__()

        self.actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, rnn_num_layers, rnn_hidden_size_actor, device,actor=True, rnn = rnn).to(device)
        self.critic = Network(in_critic, hidden_in_critic, hidden_out_critic, 1, rnn_num_layers, rnn_hidden_size_critic, device, rnn = rnn).to(device)
        # self.target_actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, rnn_num_layers, rnn_hidden_size_actor, device, actor=True, rnn = rnn).to(device)
        self.target_critic = Network(in_critic, hidden_in_critic, hidden_out_critic, 1, rnn_num_layers, rnn_hidden_size_critic, device, rnn = rnn).to(device)

        self.noise = OUNoise(out_actor, scale=1.0 )
        self.device = device
        
        # from torchsummary import summary
        
        # import pdb; pdb.set_trace()
        # summary(self.actor, (3, 224, 224))

        # initialize targets same as original networks
        # hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic, weight_decay=weight_decay)
        # self.actor_optimizer = AdamW(self.actor.parameters(), lr=lr_actor, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False)
        # self.critic_optimizer = AdamW(self.critic.parameters(), lr=lr_critic, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False)
        
        # Alpha 
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.alpha = alpha
        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        if self.automatic_entropy_tuning is True:
            self.target_entropy = -torch.prod(torch.Tensor(out_actor).to(self.device)).item()
            self.log_alpha = (torch.zeros(1, requires_grad=True, device=self.device)+np.log(self.alpha)).detach().requires_grad_(True)
            self.alpha_optimizer = Adam([self.log_alpha], lr=lr_actor)
            
            

    def act(self, his, obs, noise=0.0):
        his = his.to(self.device)
        obs = obs.to(self.device)
        if noise > 0.0:
            action, _ = self.actor.sample_normal(his,obs) 
        else:
            action, _ = self.actor.forward(his,obs) 
            action = action.cpu().clamp(-1, 1)
        # actions.cpu().detach().numpy()[0]
        return action.cpu()
    

    def act_prob(self, his, obs, noise=0.0):
        his = his.to(self.device)
        obs = obs.to(self.device)
        #before 5/12/2022
        # actions, log_probs = self.actor.sample_normal(his,obs) 
        #After 5/12/2022
        #from https://github.com/kengz/SLM-Lab/blob/dda02d00031553aeda4c49c5baa7d0706c53996b/slm_lab/agent/algorithm/sac.py
        #and https://medium.com/@kengz/soft-actor-critic-for-continuous-and-discrete-actions-eeff6f651954
        if noise > 0.0:
            action, log_probs = self.actor.sample_normal(his,obs) 
        else:
            action, log_probs = self.actor.forward(his,obs) 
            action = action.cpu().clamp(-1, 1)
        return action.cpu(), log_probs
