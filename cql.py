import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from copy import deepcopy
from math import sqrt
import matplotlib.pyplot as plt
import os

class DeepQNetwork(nn.Module):
    def __init__(self,ALPHA,init_bias):
        super(DeepQNetwork, self).__init__()
        self.fc1=nn.Linear(17,64)
        nn.init.kaiming_uniform_(self.fc1.weight,mode='fan_out',nonlinearity='relu')
        self.fc2=nn.Linear(64,32)
        nn.init.kaiming_uniform_(self.fc2.weight,mode='fan_out',nonlinearity='relu')
        self.fc3=nn.Linear(32,3)
        self.fc3.bias.data.fill_(init_bias)



        self.optimizer = optim.Adam(self.parameters(),lr=ALPHA)
        self.loss = nn.MSELoss()
        self.device= T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):
        observation = T.Tensor(observation).to(self.device)
        ##resize ?
        observation = F.relu(self.fc1(observation))
        observation = F.relu(self.fc2(observation))
        #observation = F.relu(self.fc3(observation))
        Qvalues = self.fc3(observation)

        return Qvalues

class Agent(object):
    def __init__(self,gamma,alpha,A,maxMemorySize,replace_target_cnt=None,actionSpace=[0,1,2],init_bias=0,memCntr=0,actionMemory=[]):
        self.GAMMA= gamma
        self.A= A
        self.actionSpace = actionSpace
        self.memSize = maxMemorySize
        self.steps = 0
        self.learn_step_counter = 0
        self.actionMemory = actionMemory
        self.memCntr = memCntr
        self.replace_target_cnt = replace_target_cnt
        self.Q_eval = DeepQNetwork(alpha,init_bias)
        self.Q_next = DeepQNetwork(alpha,init_bias)
        """
        self.sumR = initialSumR
        self.sumSqR = initialSumSqR
        if self.memCntr==0:

            self.sigmaSq = 0.01
        else:
            self.sigmaSq=self.calculate_sigma()
        """
        self.loss = []

    def calculate_sigma(self):
        mean = self.sumR / self.memCntr
        print(f'mean:{mean}')
        r = (self.sumSqR/self.memCntr) - mean * mean
        print(f'var:{r}')
        return r


    def storeTransition(self,state,action,reward,state2):
        if self.memCntr >= self.memSize:
            temp = deepcopy(self.actionMemory[self.memCntr%self.memSize])
            self.actionMemory[self.memCntr%self.memSize]=[state,action,reward,state2]
        else:
            temp = None
            self.actionMemory.append([state,action,reward,state2])
        self.memCntr+=1
        self.sumR+= reward
        self.sumSqR+= (reward*reward)
        self.sigmaSq=self.calculate_sigma()
        return temp

    def chooseAction(self, observation):
        actions = self.Q_eval.forward(observation)
        action = T.argmax(actions).item()

        return action

    def learn(self, batch_size):
        self.Q_eval.optimizer.zero_grad()
        if self.replace_target_cnt is not None and self.learn_step_counter % self.replace_target_cnt == 0:
            self.Q_next.load_state_dict(self.Q_eval.state_dict())

        miniBatch = random.choices(self.actionMemory,k=batch_size)
        memory = np.array(miniBatch)

        states = T.FloatTensor(list(memory[:,0][:])).to(self.Q_eval.device)
        actions = T.LongTensor(list(memory[:,1][:])).to(self.Q_eval.device)
        rewards = T.FloatTensor(list(memory[:,2][:])).to(self.Q_eval.device)
        next_states = T.FloatTensor(list(memory[:,3][:])).to(self.Q_next.device)

        actions = actions.view(actions.size(0),1)
        with torch.no_grad():
            Q_targets_next = self.Q_next(next_states).detach().max(1)[0].unsqueeze(1)
            Q_targets = rewards + self.GAMMA * Q_targets_next
        Q_a_s = self.Q_eval.forward(states)
        Q_expected= Q_a_s.gather(1,actions)

        cql_loss = torch.logsumexp(Q_a_s, dim=1).mean() - Q_expected.mean()

        bellman_error = F.mse_loss(Q_expected,Q_targets)

        loss = self.A*cql_loss + 0.5*bellman_error

        cql_loss.backward()
        self.Q_eval.optimizer.step()

        self.learn_step_counter+=1
if __name__=='__main__':
    with open('./InitialReplayMemory.npy','rb') as f:
        epochs =100
        ReplayMemory = np.load(f,allow_pickle=True).tolist()
        agent = Agent(gamma=0.95,alpha=0.001,A=5,maxMemorySize=300,
         replace_target_cnt=60,
         actionSpace=[0,1,2],init_bias=974.5715154675061,memCntr=300,actionMemory=ReplayMemory)
        print(agent.sigmaSq)
        for i in range(epochs):

            agent.learn(32)

        pwd = os.getcwd()
        filename = f"cql_300"
        path = os.path.join(pwd,filename)
        os.mkdir(path)
        f1 = path +'/Q_eval.pt'
        torch.save(agent.Q_eval.state_dict(),f1)
        f2 = path+'/Q_next.pt'
        torch.save(agent.Q_eval.state_dict(),f2)
