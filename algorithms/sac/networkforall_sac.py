import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
from torch.distributions.normal import Normal

'''
Code partially extracted from:
https://github.com/denisyarats/pytorch_sac/blob/81c5b536d3a1c5616b2531e446450df412a064fb/agent/sac.py
https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/SAC/sac_torch.py
https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/sac.py

'''
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Network(nn.Module):
    def __init__(self, input_size, hidden_in_dim, hidden_out_dim, output_dim, rnn_num_layers, rnn_hidden_size, device, actor=False, rnn=True):
        super(Network, self).__init__()
        """self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_norm.weight.data.fill_(1)
        self.input_norm.bias.data.fill_(0)"""
        self.device = device
        self.rnn_num_layers = rnn_num_layers
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_active = rnn
        self.aux_mul = 1
        self.reparam_noise = 1e-6
        # Linear NN layers
        if actor == True:
            # Recurrent NN layers (LSTM)
            if self.rnn_active:
                # self.rnn = nn.RNN(input_size, rnn_hidden_size, rnn_num_layers, batch_first=True)
                # self.rnn = nn.GRU(input_size, rnn_hidden_size, rnn_num_layers, batch_first=True)
                self.rnn = nn.LSTM(input_size, rnn_hidden_size, rnn_num_layers, batch_first=True)
                self.aux_mul = 2
            self.fc0 = nn.Linear(input_size - 1 ,rnn_hidden_size)
            self.fc1 = nn.Linear(rnn_hidden_size*self.aux_mul,hidden_in_dim)
            self.fc_mu = nn.Linear(hidden_in_dim, output_dim)
            self.fc_sigma = nn.Linear(hidden_in_dim, output_dim)
        else:
            # Recurrent NN layers (LSTM)
            # self.rnn = nn.RNN(input_size, rnn_hidden_size, rnn_num_layers, batch_first=True)
            # self.rnn = nn.GRU(input_size, rnn_hidden_size, rnn_num_layers, batch_first=True)
            #Q1
            if self.rnn_active:
                self.rnn_q1 = nn.LSTM(input_size, rnn_hidden_size, rnn_num_layers, batch_first=True)
                self.aux_mul = 2
            self.fc0_q1 = nn.Linear(input_size,rnn_hidden_size)
            self.fc1_q1 = nn.Linear(rnn_hidden_size*self.aux_mul,hidden_in_dim)
            self.fc2_q1 = nn.Linear(hidden_in_dim,output_dim)
            #Q2
            if self.rnn_active:
                self.rnn_q2 = nn.LSTM(input_size, rnn_hidden_size, rnn_num_layers, batch_first=True)
                self.aux_mul = 2
            self.fc0_q2 = nn.Linear(input_size,rnn_hidden_size)
            self.fc1_q2 = nn.Linear(rnn_hidden_size*self.aux_mul,hidden_in_dim)
            self.fc2_q2 = nn.Linear(hidden_in_dim,output_dim)     
        self.nonlin = f.relu #leaky_relu
        self.nonlin_tanh = torch.tanh #tanh
        self.actor = actor
        self.reset_parameters()
        
    def reset_parameters(self):
        if self.actor == True:
            if self.rnn_active:
                #method 1: (from https://discuss.pytorch.org/t/initializing-parameters-of-a-multi-layer-lstm/5791)
                for name, param in self.rnn.named_parameters():
                  if 'bias' in name:
                     nn.init.constant_(param, 0.0)
                  elif 'weight' in name:
                     nn.init.xavier_normal_(param)
                     
                # #method 2:
                # for layer_p in self.rnn._all_weights:
                #     for p in layer_p:
                #         if 'weight' in p:
                #             nn.init.normal_(self.rnn.__getattr__(p), 0.0, 0.02)
                            
            # self.rnn.weight.data.uniform_(*hidden_init(self.rnn))
            self.fc0.weight.data.uniform_(*hidden_init(self.fc0))
            self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
            self.fc_mu.weight.data.uniform_(*hidden_init(self.fc_mu))
            self.fc_sigma.weight.data.uniform_(*hidden_init(self.fc_sigma))
        else:
            #Q1
            if self.rnn_active:
                #method 1: (from https://discuss.pytorch.org/t/initializing-parameters-of-a-multi-layer-lstm/5791)
                for name, param in self.rnn_q1.named_parameters():
                  if 'bias' in name:
                     nn.init.constant_(param, 0.0)
                  elif 'weight' in name:
                     nn.init.xavier_normal_(param)
                     
                # #method 2:
                # for layer_p in self.rnn_q1._all_weights:
                #     for p in layer_p:
                #         if 'weight' in p:
                #             nn.init.normal_(self.rnn_q1.__getattr__(p), 0.0, 0.02)
                            
            # self.rnn.weight.data.uniform_(*hidden_init(self.rnn_q1))
            self.fc0_q1.weight.data.uniform_(*hidden_init(self.fc0_q1))
            self.fc1_q1.weight.data.uniform_(*hidden_init(self.fc1_q1))
            self.fc2_q1.weight.data.uniform_(*hidden_init(self.fc2_q1))
            #Q2
            if self.rnn_active:
                #method 1: (from https://discuss.pytorch.org/t/initializing-parameters-of-a-multi-layer-lstm/5791)
                for name, param in self.rnn.named_parameters():
                  if 'bias' in name:
                     nn.init.constant_(param, 0.0)
                  elif 'weight' in name:
                     nn.init.xavier_normal_(param)
                     
                # #method 2:
                # for layer_p in self.rnn_q2._all_weights:
                #     for p in layer_p:
                #         if 'weight' in p:
                #             nn.init.normal_(self.rnn_q2.__getattr__(p), 0.0, 0.02)
                            
            # self.rnn.weight.data.uniform_(*hidden_init(self.rnn_q2))
            self.fc0_q2.weight.data.uniform_(*hidden_init(self.fc0_q2))
            self.fc1_q2.weight.data.uniform_(*hidden_init(self.fc1_q2))
            self.fc2_q2.weight.data.uniform_(*hidden_init(self.fc2_q2))

    def forward(self, x1, x2):
        if self.actor:
            # return a vector of the force
            # RNN
            if self.rnn_active:
                h0 = torch.zeros(self.rnn_num_layers, x1.size(0), self.rnn_hidden_size).to(self.device) #Initial values for RNN
                c0 = torch.zeros(self.rnn_num_layers, x1.size(0), self.rnn_hidden_size).to(self.device) #Initial values for RNN
                # out, _ = self.rnn(x1,h0)
                out, _ = self.rnn(x1,(h0,c0))
                # out: batch_size, seq_legnth, hidden_size
                out = out[:,-1,:]
                # out: batch_size, hidden_size
                h00 = self.nonlin(self.fc0(x2))
                prob = torch.cat((out,h00), dim=1)
            else:
                prob = self.nonlin(self.fc0(x2))
            # Linear
            prob = self.nonlin(self.fc1(prob))           
            mean = self.fc_mu(prob)
            log_std = self.fc_sigma(prob)
            log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX) #from https://github.com/pranz24/pytorch-soft-actor-critic/blob/398595e0d9dca98b7db78c7f2f939c969431871a/model.py#L94
            return mean, log_std

        
        else:
            # critic network simply outputs a number       
            #Q1
            if self.rnn_active:
                # RNN
                h0_q1 = torch.zeros(self.rnn_num_layers, x1.size(0), self.rnn_hidden_size).to(self.device) #Initial values for RNN
                c0_q1 = torch.zeros(self.rnn_num_layers, x1.size(0), self.rnn_hidden_size).to(self.device) #Initial values for RNN
                # out, _ = self.rnn(x1,h0)
                out_q1, _ = self.rnn_q1(x1,(h0_q1,c0_q1))
                # out: batch_size, seq_legnth, hidden_size
                out_q1 = out_q1[:,-1,:]
                # out: batch_size, hidden_size
                h00_q1 = self.nonlin(self.fc0_q1(x2))
                x_q1 = torch.cat((out_q1,h00_q1), dim=1)
            else:
                x_q1 = self.nonlin(self.fc0_q1(x2))
            # Linear
            h1_q1 = self.nonlin(self.fc1_q1(x_q1))       
            h2_q1 = (self.fc2_q1(h1_q1))
            
            #Q2
            if self.rnn_active:
                # RNN
                h0_q2 = torch.zeros(self.rnn_num_layers, x1.size(0), self.rnn_hidden_size).to(self.device) #Initial values for RNN
                c0_q2 = torch.zeros(self.rnn_num_layers, x1.size(0), self.rnn_hidden_size).to(self.device) #Initial values for RNN
                # out, _ = self.rnn(x1,h0)
                out_q2, _ = self.rnn_q2(x1,(h0_q2,c0_q2))
                # out: batch_size, seq_legnth, hidden_size
                out_q2 = out_q2[:,-1,:]
                # out: batch_size, hidden_size
                h00_q2 = self.nonlin(self.fc0_q2(x2))
                x_q2 = torch.cat((out_q2,h00_q2), dim=1)
            else:
                x_q2 = self.nonlin(self.fc0_q2(x2))
            # Linear
            h1_q2 = self.nonlin(self.fc1_q2(x_q2))       
            h2_q2 = (self.fc2_q2(h1_q2))
            
            return h2_q1, h2_q2
        
        
    def sample_normal(self, x1, x2):
        mean, log_std = self.forward(x1,x2)
        std = log_std.exp()
        
        normal = Normal(mean,std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action = torch.tanh(x_t) # for squashed Gaussian distribution (which means that is bounded between -1 and 1)
                
        log_prob = normal.log_prob(x_t)
        #Enforcing Action Bound
        log_prob -= torch.log(1-action.pow(2)+self.reparam_noise) #done as https://arxiv.org/pdf/2007.14430.pdf
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob
    


