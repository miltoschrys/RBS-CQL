

import numpy as np
import time
import matplotlib.pyplot as plt
from copy import deepcopy
import sys
import os
import torch
import pickle

if __name__ =='__main__':
    """
    with open('agent.txt','rb') as f:
        agent= pickle.load(f)
    """
    with open('/exploration200/experiencedataset.npy','rb'),open('/exploration101/experiencedataset.npy') as f1,f2:
        MemoryDataset = np.load(f,allow_pickle=True)
        for m in MemoryDataset:
            print("start state")
            print(m[0])
            print("action")
            print(m[1])
            print("reward")
            print(m[2])
            print("end state")
            print(m[3])
"""
    print(MemoryDataset)
    attrs = vars(agent)
    for it in attrs.items():
        print(it)
"""
