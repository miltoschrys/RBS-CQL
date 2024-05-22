import numpy as np
import time
import matplotlib.pyplot as plt
from copy import deepcopy
import sys
import os
import torch
import pickle
import random
from ddqn import DeepQNetwork, Agent
import torch

if __name__ =='__main__':

    with open('./training40/experiencedataset.npy','rb') as f1:
        replaymemory = np.load(f1,allow_pickle=True)
        print(replaymemory.shape)
        print(replaymemory)
    with open('./training40/Qeval.pt','rb') as f2:
        Qeval = DeepQNetwork(0.001,0)
        Qeval.load_state_dict(torch.load(f2))
        print(Qeval.state_dict())
    with open('./training40/agent.txt','rb') as f3:
        agent = pickle.load(f3)
        print(vars(agent))
