# individual network settings for each actor + critic pair
# see networkforall for details

from networkforall import Network
from utilities import hard_update, gumbel_softmax, onehot_from_logits
from torch.optim import Adam, AdamW
import torch
import numpy as np


# add OU noise for exploration
from OUNoise import OUNoise

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

class DDPGAgent():
    def __init__(self, in_actor, hidden_in_actor, hidden_out_actor, out_actor, in_critic, hidden_in_critic, hidden_out_critic, rnn_num_layers, rnn_hidden_size_actor, rnn_hidden_size_critic , lr_actor=1.0e-2, lr_critic=1.0e-2, weight_decay=1.0e-5, device = 'cpu', rnn=True):
        super(DDPGAgent, self).__init__()

        self.actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, rnn_num_layers, rnn_hidden_size_actor, device,actor=True,rnn=rnn).to(device)
        self.critic = Network(in_critic, hidden_in_critic, hidden_out_critic, 1, rnn_num_layers, rnn_hidden_size_critic, device, rnn=rnn).to(device)
        self.target_actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, rnn_num_layers, rnn_hidden_size_actor, device, actor=True, rnn=rnn).to(device)
        self.target_critic = Network(in_critic, hidden_in_critic, hidden_out_critic, 1, rnn_num_layers, rnn_hidden_size_critic, device, rnn=rnn).to(device)

        self.noise = OUNoise(out_actor, scale=1.0 )
        self.device = device
        
        # from torchsummary import summary
        
        # import pdb; pdb.set_trace()
        # summary(self.actor, (3, 224, 224))

        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic, weight_decay=weight_decay)
        # self.actor_optimizer = AdamW(self.actor.parameters(), lr=lr_actor, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False)
        # self.critic_optimizer = AdamW(self.critic.parameters(), lr=lr_critic, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False)



    def act(self, his, obs, noise=0.0):
        his = his.to(self.device)
        obs = obs.to(self.device)
        action = self.actor(his,obs).cpu() + noise*self.noise.noise()
        action = action.clamp(-1, 1)
        return action

    def target_act(self, his, obs, noise=0.0):
        his = his.to(self.device)
        obs = obs.to(self.device)
        action = self.target_actor(his,obs).cpu() + noise*self.noise.noise()
        action = action.clamp(-1, 1)
        return action
