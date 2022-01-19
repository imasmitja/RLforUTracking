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
            self.fc2 = nn.Linear(hidden_in_dim,output_dim)
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
            # self.rnn.weight.data.uniform_(*hidden_init(self.rnn))
            self.fc0.weight.data.uniform_(*hidden_init(self.fc0))
            self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
            self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        else:
            #Q1
            # self.rnn_q1.weight.data.uniform_(*hidden_init(self.rnn_q1))
            self.fc0_q1.weight.data.uniform_(*hidden_init(self.fc0_q1))
            self.fc1_q1.weight.data.uniform_(*hidden_init(self.fc1_q1))
            self.fc2_q1.weight.data.uniform_(*hidden_init(self.fc2_q1))
            #Q2
            # self.rnn_q2.weight.data.uniform_(*hidden_init(self.rnn_q2))
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
        
    def Q1(self, x1, x2):
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
        return h2_q1
        
    


