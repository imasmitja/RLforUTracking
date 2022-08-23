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
# RNN = 'GRU'
RNN = 'LSTM'

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
        self.reparam_noise = 1e-6
        self.input_size = input_size
        
        # print('NETWORK: inputsize=%i hiddensize=%i hiddenin=%i'%(input_size - 1,rnn_hidden_size,hidden_in_dim))
        # Linear NN layers
        if actor == True:
            self.fc0 = nn.Linear(input_size - 1 ,rnn_hidden_size)
            # Recurrent NN layers (LSTM)
            if RNN == 'GRU':
                self.rnn = nn.GRU(rnn_hidden_size, rnn_hidden_size, rnn_num_layers, batch_first=True)
            if RNN == 'LSTM':
                self.rnn = nn.LSTM(rnn_hidden_size, rnn_hidden_size, rnn_num_layers, batch_first=True)
            self.fc1 = nn.Linear(rnn_hidden_size,hidden_in_dim)
            self.fc_mu = nn.Linear(hidden_in_dim, output_dim)
            self.fc_sigma = nn.Linear(hidden_in_dim, output_dim)
            self.h0 = []
            self.c0 = []
        else:
            # Recurrent NN layers (LSTM)
            # self.rnn = nn.RNN(input_size, rnn_hidden_size, rnn_num_layers, batch_first=True)
            # self.rnn = nn.GRU(input_size, rnn_hidden_size, rnn_num_layers, batch_first=True)
            #Q1
            self.fc0_q1 = nn.Linear(input_size,rnn_hidden_size)
            if RNN == 'GRU':
                self.rnn_q1 = nn.GRU(rnn_hidden_size, rnn_hidden_size, rnn_num_layers, batch_first=True)
            if RNN == 'LSTM':
                self.rnn_q1 = nn.LSTM(rnn_hidden_size, rnn_hidden_size, rnn_num_layers, batch_first=True)
            self.fc1_q1 = nn.Linear(rnn_hidden_size,hidden_in_dim)
            self.fc2_q1 = nn.Linear(hidden_in_dim,output_dim)
            self.h0_q1 = []
            self.c0_q1 = []
            #Q2
            self.fc0_q2 = nn.Linear(input_size,rnn_hidden_size)
            if RNN == 'GRU':
                self.rnn_q2 = nn.GRU(rnn_hidden_size, rnn_hidden_size, rnn_num_layers, batch_first=True)
            if RNN == 'LSTM':
                self.rnn_q2 = nn.LSTM(rnn_hidden_size, rnn_hidden_size, rnn_num_layers, batch_first=True)
            self.fc1_q2 = nn.Linear(rnn_hidden_size,hidden_in_dim)
            self.fc2_q2 = nn.Linear(hidden_in_dim,output_dim)
            self.h0_q2 = []
            self.c0_q2 = []
        self.nonlin = f.relu #leaky_relu
        self.nonlin_tanh = torch.tanh #tanh
        self.actor = actor
        
        self.reset_parameters()
        
    def reset_parameters(self):
        if self.actor == True:
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
            self.h0 = torch.zeros(self.rnn_num_layers, 1, self.rnn_hidden_size).to(self.device) #Initial values for RNN
            self.c0 = torch.zeros(self.rnn_num_layers,1, self.rnn_hidden_size).to(self.device) #Initial values for RNN
                
        else:
            #Q1
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
                            
            # self.rnn_q1.weight.data.uniform_(*hidden_init(self.rnn_q1))
            self.fc0_q1.weight.data.uniform_(*hidden_init(self.fc0_q1))
            self.fc1_q1.weight.data.uniform_(*hidden_init(self.fc1_q1))
            self.fc2_q1.weight.data.uniform_(*hidden_init(self.fc2_q1))
            self.h0_q1 = torch.zeros(self.rnn_num_layers, 1, self.rnn_hidden_size).to(self.device) #Initial values for RNN
            self.c0_q1 = torch.zeros(self.rnn_num_layers, 1, self.rnn_hidden_size).to(self.device) #Initial values for RNN
                
            #Q2
            #method 1: (from https://discuss.pytorch.org/t/initializing-parameters-of-a-multi-layer-lstm/5791)
            for name, param in self.rnn_q2.named_parameters():
              if 'bias' in name:
                 nn.init.constant_(param, 0.0)
              elif 'weight' in name:
                 nn.init.xavier_normal_(param)
                 
            # #method 2:
            # for layer_p in self.rnn_q2._all_weights:
            #     for p in layer_p:
            #         if 'weight' in p:
            #             nn.init.normal_(self.rnn_q2.__getattr__(p), 0.0, 0.02)
                            
            # self.rnn_q2.weight.data.uniform_(*hidden_init(self.rnn_q2))
            self.fc0_q2.weight.data.uniform_(*hidden_init(self.fc0_q2))
            self.fc1_q2.weight.data.uniform_(*hidden_init(self.fc1_q2))
            self.fc2_q2.weight.data.uniform_(*hidden_init(self.fc2_q2))
            self.h0_q2 = torch.zeros(self.rnn_num_layers, 1, self.rnn_hidden_size).to(self.device) #Initial values for RNN
            self.c0_q2 = torch.zeros(self.rnn_num_layers, 1, self.rnn_hidden_size).to(self.device) #Initial values for RNN
                

    def forward(self, x1, x2):
        if self.actor:
            # return a vector of the force
            # RNN
            h00 = self.nonlin(self.fc0(x2))
            # out, _ = self.rnn(x1,h0)
            # import pdb; pdb.set_trace()
            if RNN=='LSTM':
                out, (hn, cn) = self.rnn(h00.reshape(1,x2.shape[0],self.rnn_hidden_size),(self.h0,self.c0))
            if RNN=='GRU':
                out, hn = self.rnn(h00.reshape(1,x2.shape[0],self.rnn_hidden_size),self.h0)
            # out: batch_size, seq_legnth, hidden_size
            prob = out.reshape(x2.shape[0],self.rnn_hidden_size)
            # Linear
            prob = self.nonlin(self.fc1(prob))           
            mean = self.fc_mu(prob)
            log_std = self.fc_sigma(prob)
            log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX) #from https://github.com/pranz24/pytorch-soft-actor-critic/blob/398595e0d9dca98b7db78c7f2f939c969431871a/model.py#L94
            #copy history
            self.h0 = hn.detach()
            if RNN=='LSTM':
                self.c0 = cn.detach()
            return mean, log_std

        
        else:
            # critic network simply outputs a number       
            #Q1
            h00_q1 = self.nonlin(self.fc0_q1(x2))
            # out, _ = self.rnn(x1,h0)
            if RNN=='LSTM':
                out_q1, (hn_q1,cn_q1) = self.rnn_q1(h00_q1.reshape(1,x2.shape[0],self.rnn_hidden_size),(self.h0_q1,self.c0_q1))
            if RNN=='GRU':
                out_q1, hn_q1 = self.rnn_q1(h00_q1.reshape(1,x2.shape[0],self.rnn_hidden_size),self.h0_q1)
            # out: batch_size, seq_legnth, hidden_size
            x_q1 = out_q1.reshape(x2.shape[0],self.rnn_hidden_size)
            # Linear
            h1_q1 = self.nonlin(self.fc1_q1(x_q1))       
            h2_q1 = (self.fc2_q1(h1_q1))
            #copy history
            self.h0_q1 = hn_q1.detach()
            if RNN=='LSTM':
                self.c0_q1 = cn_q1.detach()
            
            #Q2
            h00_q2 = self.nonlin(self.fc0_q2(x2))
            # RNN
            # out, _ = self.rnn(x1,h0)
            if RNN=='LSTM':
                out_q2, (hn_q2,cn_q2) = self.rnn_q2(h00_q2.reshape(1,x2.shape[0],self.rnn_hidden_size),(self.h0_q2,self.c0_q2))
            if RNN=='GRU':
                out_q2, hn_q2 = self.rnn_q2(h00_q2.reshape(1,x2.shape[0],self.rnn_hidden_size),self.h0_q2)
            # out: batch_size, seq_legnth, hidden_size
            x_q2 = out_q2.reshape(x2.shape[0],self.rnn_hidden_size)
            # Linear
            h1_q2 = self.nonlin(self.fc1_q2(x_q2))       
            h2_q2 = (self.fc2_q2(h1_q2))
            #copy history
            self.h0_q2 = hn_q2.detach()
            if RNN=='LSTM':
                self.c0_q2 = cn_q2.detach()
            
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
    


