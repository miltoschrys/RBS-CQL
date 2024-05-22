from ddqn import DeepQNetwork, Agent
#from utils import plotLearning
from featureloader import ParameterLoader
from cluster import cluster
import numpy as np
import time
import matplotlib.pyplot as plt
from copy import deepcopy
import sys
import os
import torch
import pickle
if __name__ =='__main__':
    agent= Agent(gamma=0.95,epsilon=1.0,alpha=0.001,maxMemorySize=300,replace_target_cnt=300)
    MemoryDataset = [[0,0,0,0],[1,1,1,1],[2,2,2,2]]
    pwd = os.getcwd()
    steps = 1000
    filename = f"modelmem{steps}"
    path = os.path.join(pwd,filename)
    os.mkdir(path)
    f1 = path +'/Qeval.pt'
    torch.save(agent.Q_eval.state_dict(),f1)
    f2 = path+'/Q_next.pt'
    torch.save(agent.Q_eval.state_dict(),f2)
    with open(path+'/experiencedataset.npy','wb') as f:
        np.save(f,np.asarray(MemoryDataset))
    with open(path+'/agent.txt','wb') as f:
        pickle.dump(agent,f)

    """
    if sys.argv[1] is not None and sys.argv[2] is not None:
        agent= Agent(gamma=0.95,epsilon=1.0,alpha=0.001,maxMemorySize=300,replace_target_cnt=300)
        agent.load_state_dict(torch.load(sys.argv[1]))



    else:
        agent= Agent(gamma=0.95,epsilon=1.0,alpha=0.001,maxMemorySize=300,replace_target_cnt=300)
        """
