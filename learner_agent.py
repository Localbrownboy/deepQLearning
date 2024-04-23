import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from deep_q_net import DeepQNetwork
class Agent():
    def __init__(self , gamma, epsilon , lr, input_dims, batch_size, n_actions , max_mem_size = 100000, eps_min = 0.01 , eps_dec =5e-6  ):
        self.gamma = gamma 
        self.epsilon = epsilon
        self.lr = lr
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.mem_size = max_mem_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.action_space = [i for i in range(n_actions)]
        self.mem_cntr = 0
        
        self.q_eval = DeepQNetwork(lr=self.lr , input_dims=self.input_dims , fc1_dims=256 , fc2_dims=256, n_actions=n_actions)

        self.state_memory = np.zeros( (self.mem_size, *self.input_dims) , dtype=np.float32)
        self.new_state_memory = np.zeros( (self.mem_size, *self.input_dims) , dtype=np.float32)
        self.action_memory = np.zeros( (self.mem_size, *self.input_dims) , dtype=np.int32)
        self.reward_memory = np.zeros( (self.mem_size, *self.input_dims) , dtype=np.float32)
        self.terminal_memory = np.zeros( (self.mem_size, *self.input_dims) , dtype=bool)

    def store_transition(self, state, action, reward, state_ , done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] =state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr+=1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.q_eval.device)
            actions = self.q_eval.forward(state=state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action
    

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return
        self.q_eval.optimizer.zero_grad() # its been cached

        max_mem = min(self.mem_cntr , self.mem_size)

        batch = np.random.choice(max_mem , self.batch_size , replace=False) 
        batch_index = np.arange(self.batch_size , dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.q_eval.device)
        action_batch = self.action_memory[batch]

        q_eval = self.q_eval.forward(state_batch)[batch_index , action_batch]
        q_next = self.q_eval.forward(new_state_batch) ## can replace this with a target network
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma*max(q_next , dim=1)[0]  # is the hardcoded max value of the next state 

        loss = self.q_eval.loss(q_target , q_eval).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()

        if self.eps_min < self.epsilon - self.eps_dec:
            self.epsilon -= self.eps_dec
        else:
            self.epsilon = self.eps_min
    










