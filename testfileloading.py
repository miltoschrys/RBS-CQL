#from ddqn import DeepQNetwork, Agent
#from utils import plotLearning

import numpy as np
import time
import matplotlib.pyplot as plt
from copy import deepcopy
import sys
import os
import torch
import pickle
import random

if __name__ =='__main__':

    with open('./exploration2-301/experiencedataset.npy','rb') as f1:
        MemoryDataset = np.load(f1,allow_pickle=True)
        """
        print(len(MemoryDataset1))
        print(MemoryDataset1.shape)
        print(len(MemotyDataset2))
        print(MemotyDataset2.shape)
        """
        #MemoryDataset=np.concatenate((MemoryDataset1,MemotyDataset2),axis=0)
        """
        print(len(MemoryDataset))
        print(MemoryDataset.shape)
        print(MemoryDataset[0,:])
        """
        #MemoryDataset=np.delete(MemoryDataset,(0),axis=0)
        """
        print(MemoryDataset.shape)
        print(MemoryDataset[0,:])
        #print(MemoryDataset[:,2])
        """
        mean = np.sum(MemoryDataset[:,2]) / float(MemoryDataset.shape[0])
        print(f'total mean: {mean}')

        sumsq = np.sum(np.square(MemoryDataset[:,2])) / float(MemoryDataset.shape[0])
    #    sumsq = np.true_divide((np.sum(map((lambda a: a**a),MemoryDataset[:,2])) ), float(MemoryDataset.shape[0]))
        print(f'squares total sum: {sumsq}')
        var = sumsq - mean*mean
        print(f'total var: {var}')
        miniBatchl = random.choices(MemoryDataset[:,2].tolist(),k=32)
        miniBatch = np.array(miniBatchl)
        mean = np.sum(miniBatch) / 32.0
        print(f'miniBatch mean: {mean}')
        sumsq = np.sum(np.square(miniBatch)) / 32.0
        print(f'squares miniBatch sum: {sumsq}')
        var = sumsq - mean*mean
        print(f'miniBatch var: {var}')
        through = np.array(MemoryDataset[:,0].tolist())[:,4]
        late = np.array(MemoryDataset[:,0].tolist())[:,3]
        meanthroughput = np.mean(through)
        meanlat = np.mean(late)
        print(f'mean throughput{meanthroughput*1500}')
        print(f'mean latency{meanlat*1}')
        vms = np.array(MemoryDataset[:,0].tolist())[:,0]
        meanvms = np.mean(vms)
        print(f'mean vms: {meanvms*3 + 5.0}')
        vm5s = vms==0
        #print(MemoryDataset[vm5s,0])
        through5s = np.array(MemoryDataset[vm5s,0].tolist())[:,4]
        meanthrough5s = np.mean(through5s)
        varthrough5s = np.var(through5s)
        print(f"mean through for 5s: {meanthrough5s*1500}")
        print(f"var through for 5s: {varthrough5s*1500}")
        vm6s = vms==(1/3)
        through6s = np.array(MemoryDataset[vm6s,0].tolist())[:,4]
        meanthrough6s = np.mean(through6s)
        varthrough6s = np.var(through6s)
        print(f"mean through for 6s: {meanthrough6s*1500}")
        print(f"var through for 6s: {varthrough6s*1500}")
        vm7s = vms==(2/3)
        through7s = np.array(MemoryDataset[vm7s,0].tolist())[:,4]
        meanthrough7s = np.mean(through7s)
        varthrough7s = np.var(through7s)
        print(f"mean through for 7s: {meanthrough7s*1500}")
        print(f"var through for 7s: {varthrough7s*1500}")
        vm8s = vms==1
        through8s = np.array(MemoryDataset[vm8s,0].tolist())[:,4]
        meanthrough8s = np.mean(through8s)
        varthrough8s = np.var(through8s)
        print(f"mean through for 8s: {meanthrough8s*1500}")
        print(f"var through for 8s: {varthrough8s*1500}")
        #maxthrough = { 5:600, 6:800, 7:1100, 8:2000}
        #MemoryDataset[:,2] = min(0.01*(through[:]*1500)  -(vms[:]*3 +5.0), maxthrough[(vms[:]*3+5)])
        #print("new rewards")
        #print(MemoryDataset[:,2])
        print(MemoryDataset[0,0])
        print(MemoryDataset.shape)
        for i in range(MemoryDataset.shape[0]):
            a = np.delete(MemoryDataset[i,0],8)
            b = np.delete(MemoryDataset[i,3],8)
            MemoryDataset[i,0] = a
            MemoryDataset[i,0][9] = 0.5 + np.random.normal(0.0,1.0)
            MemoryDataset[i,3] = b
            MemoryDataset[i,3][9] = 0.5 + np.random.normal(0.0,1.0)

        print(f'new mean r: {np.mean(MemoryDataset[:,2])}')
        print(f'new gain: {np.mean(MemoryDataset[:,2]/(1-0.99))}')
        print(f'initsumR: {np.sum(MemoryDataset[:,2])}')
        print(f'initialSumSqR: {np.sum(np.square(MemoryDataset[:,2]))}')
        print(len(MemoryDataset))

        pwd = os.getcwd()
        filename = f"InitialReplayMemory"
        with open(pwd+'/InitialReplayMemory2.npy','wb') as f:
            np.save(f,MemoryDataset)
        with open('./InitialReplayMemory2.npy','rb') as f3:
            MemoryDatasetFinal = np.load(f3,allow_pickle=True)
            print(f'new mean r: {np.mean(MemoryDatasetFinal[:,2])}')
            print(f'new gain: {np.mean(MemoryDatasetFinal[:,2]/(1-0.99))}')
            print(f'initsumR: {np.sum(MemoryDatasetFinal[:,2])}')
            print(f'initialSumSqR: {np.sum(np.square(MemoryDatasetFinal[:,2]))}')
            print(MemoryDatasetFinal)
        
