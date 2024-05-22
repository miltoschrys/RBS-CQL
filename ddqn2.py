import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from copy import deepcopy
from math import sqrt
import matplotlib.pyplot as plt

class DeepQNetwork(nn.Module):
    def __init__(self,ALPHA,init_bias):
        super(DeepQNetwork, self).__init__()
        self.fc1=nn.Linear(13,48)
        nn.init.kaiming_uniform_(self.fc1.weight,mode='fan_out',nonlinearity='relu')
        self.fc2=nn.Linear(48,24)
        nn.init.kaiming_uniform_(self.fc2.weight,mode='fan_out',nonlinearity='relu')
        self.fc3=nn.Linear(24,3)
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
    def __init__(self,gamma,epsilon,alpha,maxMemorySize, initialSumR=1,initialSumSqR=1,epsEnd=0.1,replace_target_cnt=None,actionSpace=[0,1,2],init_bias=0,memCntr=0,actionMemory=[]):
        self.GAMMA= gamma
        self.EPSILON = epsilon
        self.EPS_END = epsEnd
        self.actionSpace = actionSpace
        self.memSize = maxMemorySize
        self.steps = 0
        self.learn_step_counter = 0
        self.actionMemory = actionMemory
        self.memCntr = memCntr
        self.replace_target_cnt = replace_target_cnt
        self.Q_eval = DeepQNetwork(alpha,init_bias)
        self.Q_next = DeepQNetwork(alpha,init_bias)
        self.sumR = initialSumR
        self.sumSqR = initialSumSqR
        if self.memCntr==0:

            self.sigmaSq = 0.01
        else:
            self.sigmaSq=self.calculate_sigma()
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
        rand = np.random.random()
        actions = self.Q_eval.forward(observation)
        if rand <1 - self.EPSILON:
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.actionSpace)
        self.steps += 1
        return action

    def learn(self, batch_size):
        self.Q_eval.optimizer.zero_grad()
        if self.replace_target_cnt is not None and self.learn_step_counter % self.replace_target_cnt == 0:
            self.Q_next.load_state_dict(self.Q_eval.state_dict())
        #maybe make random picks to create the mini batch
        miniBatch = random.choices(self.actionMemory,k=batch_size)
        memory = np.array(miniBatch)
        states = T.FloatTensor(list(memory[:,0][:])).to(self.Q_eval.device)
        actions = T.LongTensor(list(memory[:,1][:])).to(self.Q_eval.device)
        rewards = T.FloatTensor(list(memory[:,2][:])).to(self.Q_eval.device)
        next_states = T.FloatTensor(list(memory[:,3][:])).to(self.Q_eval.device)

        actions = actions.view(actions.size(0),1)

        curr_Q = self.Q_eval.forward(states).gather(1,actions)
        print("CURRq")
        print(curr_Q)
        next_Q = self.Q_next.forward(next_states)
        max_next_Q = T.max(next_Q,1)[0]
        max_next_Q = max_next_Q.view(max_next_Q.size(0),1)
        expected_Q = rewards + self.GAMMA * max_next_Q


        maxA= T.argmax(next_Q,dim=1).to(self.Q_next.device)
        print("max action")
        print(maxA)
        print("rewards")
        print(rewards)
        #calculate batch variance and compare with overall for return-based scaling
        mean = np.mean(memory[:,2])
        batchsigmasq = np.mean(np.square(memory[:,2])) - (mean*mean)
        #sq root of sigma for rescalling
        sigma = sqrt(max(self.sigmaSq,batchsigmasq))
        print(f'var for batch:{sigma*sigma}')

        print("Qnext")
        print(next_Q)
        print("Qnext max")
        print(T.max(next_Q,dim=1)[0])
        print("Qnext diff dim")

        print(T.mul(max_next_Q,self.GAMMA))

        print("new expected Qtarget ")
        print(expected_Q)
        expected_Q = T.div(expected_Q,sigma)

        print("normalized Qtarget")
        print(expected_Q)
        print(curr_Q)

        curr_Q = T.div(curr_Q,sigma)

        print(curr_Q)

        if self.EPSILON -2e-3 > self.EPS_END:
            self.EPSILON -= 2e-3
        else:
            self.EPSILON = self.EPS_END
        loss = self.Q_eval.loss(curr_Q,expected_Q.detach()).to(self.Q_eval.device)

        print("loss")
        print(loss)

        self.loss.append(loss.item())
        loss.backward()
        self.Q_eval.optimizer.step()
        ##print(self.Q_next.state_dict())
        self.learn_step_counter+=1
if __name__=='__main__':
    with open('./InitialReplayMemory.npy','rb') as f:
        ReplayMemory = np.load(f,allow_pickle=True).tolist()
        agent = Agent(gamma=0.95,epsilon=0,alpha=0.001,maxMemorySize=300,
         initialSumR=14618.572732012606,initialSumSqR=844308.5214899884,epsEnd=0,replace_target_cnt=30,
         actionSpace=[0,1,2],init_bias=974.5715154675061,memCntr=300,actionMemory=ReplayMemory)
        print(agent.sigmaSq)
        for i in range(1):
            t=agent.storeTransition(np.array([6.66666667e-01, 4.01093719e-01, 2.83321637e-01, 3.17994285e-02,
                                                4.70121936e-01, 6.78560661e-04, 1.00000000e+00, 5.58655925e-01,
                                                1.00000000e+00, 1.43444837e+00, 2.29875369e-02, 9.08887719e+02,
                                                8.82778143e-01]),
                                                2,45.97531860245614,
                                                np.array([1.00000000e+00, 4.01093719e-01, 4.39403509e-01, 3.10598609e-02,
                                                4.79253555e-01, 4.98717372e-04, 1.00000000e+00, 5.10876762e-01,
                                                1.00000000e+00, 1.88865095e+00, 3.78377363e-02, 6.79971930e+02,
                                                8.80489805e-01]))
            ##print(agent.actionMemory[1:11])
            agent.learn(2)


        x= np.arange(0,400,1.0)
        fig,ax1 = plt.subplots()
        color= 'r'
        ax1.set_xlabel('steps')
        ax1.set_ylabel('loss',color=color)
        ax1.plot(x,agent.loss,color=color)
        ax1.tick_params(axis='y',labelcolor=color)



        plt.show()
