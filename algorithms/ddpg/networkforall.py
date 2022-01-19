import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
# from torchsummary import summary

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
        if self.rnn_active:
            # Recurrent NN layers (LSTM)
            # self.rnn = nn.RNN(input_size, rnn_hidden_size, rnn_num_layers, batch_first=True)
            # self.rnn = nn.GRU(input_size, rnn_hidden_size, rnn_num_layers, batch_first=True)
            self.rnn = nn.LSTM(input_size, rnn_hidden_size, rnn_num_layers, batch_first=True)
            self.aux_mul = 2
        
        # Linear NN layers
        if actor == True:
            self.fc1 = nn.Linear(rnn_hidden_size*self.aux_mul,hidden_in_dim)
            self.fc0 = nn.Linear(input_size - 1 ,rnn_hidden_size)
        else:
            self.fc1 = nn.Linear(rnn_hidden_size*self.aux_mul,hidden_in_dim)
            self.fc0 = nn.Linear(input_size,rnn_hidden_size)        
        self.fc2 = nn.Linear(hidden_in_dim,output_dim)
        
        self.nonlin = f.relu #leaky_relu
        self.nonlin_tanh = torch.tanh #tanh
        self.actor = actor
        self.reset_parameters()

    def reset_parameters(self):
        # self.rnn.weight.data.uniform_(*hidden_init(self.rnn))
        self.fc0.weight.data.uniform_(*hidden_init(self.fc0))
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))

    def forward(self, x1, x2):
        if self.actor:
            # return a vector of the force
            if self.rnn_active:
                # RNN
                h0 = torch.zeros(self.rnn_num_layers, x1.size(0), self.rnn_hidden_size).to(self.device) #Initial values for RNN
                c0 = torch.zeros(self.rnn_num_layers, x1.size(0), self.rnn_hidden_size).to(self.device) #Initial values for RNN
                # out, _ = self.rnn(x1,h0)
                out, _ = self.rnn(x1,(h0,c0))
                # out: batch_size, seq_legnth, hidden_size
                out = out[:,-1,:]
                # out: batch_size, hidden_size
                h00 = self.nonlin(self.fc0(x2))
                x = torch.cat((out,h00), dim=1)
            else:
                x = self.nonlin(self.fc0(x2))
            # Linear
            h1 = self.nonlin(self.fc1(x))
            # h2 = (self.fc2(h1))
            h2 = self.nonlin_tanh(self.fc2(h1))
            return h2
            
            # # h2 is a 2D vector (a force that is applied to the agent)
            # # we bound the norm of the vector to be between 0 and 10
            # norm = torch.norm(h2)
            # # return 10.0*(torch.tanh(norm))*h2/norm if norm > 0 else 10*h2
            # return 1.0*(torch.tanh(norm))*h2/norm if norm > 0 else 1*h2
        
        else:
            # critic network simply outputs a number
            if self.rnn_active:
                # RNN
                h0 = torch.zeros(self.rnn_num_layers, x1.size(0), self.rnn_hidden_size).to(self.device) #Initial values for RNN
                c0 = torch.zeros(self.rnn_num_layers, x1.size(0), self.rnn_hidden_size).to(self.device) #Initial values for RNN
                # out, _ = self.rnn(x1,h0)
                out, _ = self.rnn(x1,(h0,c0))
                # out: batch_size, seq_legnth, hidden_size
                out = out[:,-1,:]
                # out: batch_size, hidden_size
                h00 = self.nonlin(self.fc0(x2))
                x = torch.cat((out,h00), dim=1)
            else:
                x = self.nonlin(self.fc0(x2))
            # Linear
            h1 = self.nonlin(self.fc1(x))       
            h2 = (self.fc2(h1))
            return h2


