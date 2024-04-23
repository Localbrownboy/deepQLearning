import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims , fc1_dims , fc2_dims, n_actions):
        super(DeepQNetwork , self).__init__()

        self.input_dims = input_dims
        self.lr = lr
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        
        self.fc1 = nn.Linear(*self.input_dims , self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims , self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims , self.n_actions) # output layer

        self.optimizer = optim.Adam(params=self.parameters() , lr=lr)
        self.loss = nn.MSELoss() # means squared error loss 

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self, state):
        x = T.tanh(self.fc1(state))
        x = T.tanh(self.fc2(x))
        return self.fc3(x) # this is actions estimate from last layer









