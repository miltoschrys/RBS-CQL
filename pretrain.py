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


def calculate_reward(throughput,vms):
    if vms == 5:
        return min(500.0,(0.01*throughput))
    if vms == 6:
        return min(750.0,(0.01*throughput -1.0))
    if vms == 7:
        return min(1000.0,(0.01*throughput -2.0))
    if vms == 8:
        return 0.01*throughput - 3.0
if __name__ =='__main__':

    try:

        agent= Agent(gamma=0.99,epsilon=1.0,alpha=0.001,maxMemorySize=300,replace_target_cnt=50)
        MemoryDataset = []
        env = cluster(8,5,8)
        paramLoader= ParameterLoader(env.get_cluster_size(),minstate=5,maxstate=8)
        exploration=0
        if(paramLoader.get_parameters(env.get_cluster_size(),1,'s') ==1):
            print("error")
            exit()
        #paramLoader.transform_parameters()
        paramLoader.normalize_parameters()
        observation = paramLoader.parameters_to_nparray()
        steps=0
        while steps <=300:
            print("step: "+str(steps))
            print("choosing action")
            action = agent.chooseAction(observation)
            if(action==1):
                print("not changing cluster")
            if(action==0):
                print("decrementing cluster")
            if(action==2):
                print("incrementing cluster")
            window=env.execute(action)

            while(not (env.check_operator_rediness())):
                try:
                    print("waiting for cass-operator to finish scaling")
                    time.sleep(30)
                except KeyboardInterrupt:
                    raise
            while (paramLoader.get_parameters(env.get_cluster_size(),action,window) == 1):
                print("prometheus error. waiting..")
                env.restart_metrics_collector()
                time.sleep(45)
                paramLoader= ParameterLoader(state=env.state,minstate=5,maxstate=8)
                time.sleep(320)
                ###################################
        #    time.sleep(180)# wait 2:30 min to gather metrics from the new state

            reward_dict=deepcopy(paramLoader.reward_dict())
            print(reward_dict)
            paramLoader.normalize_parameters()
            observation_=paramLoader.parameters_to_nparray()
            print(observation_)
            #reward=calculate_reward(reward_dict['throughput'],reward_dict['vms'])
            reward = 0.01*reward_dict['throughput'] - (reward_dict['vms']-5.0)
            obs=agent.storeTransition(observation,action,reward,observation_)
            if obs is not None:
                MemoryDataset.append(obs)
            exploration+=1
            observation=observation_
            steps+=1
        pwd = os.getcwd()
        filename = f"exploration2-{exploration}"
        path = os.path.join(pwd,filename)
        os.mkdir(path)
        f1 = path +'/Qeval.pt'
        torch.save(agent.Q_eval.state_dict(),f1)
        f2 = path+'/Q_next.pt'
        torch.save(agent.Q_eval.state_dict(),f2)
        with open(path+'/experiencedataset.npy','wb') as f:
            np.save(f,np.asarray(agent.actionMemory))
        with open(path+'/agent.txt','wb') as f:
            pickle.dump(agent,f)
    except Exception as e:
        pwd = os.getcwd()
        filename = f"exploration2-{exploration}"
        path = os.path.join(pwd,filename)
        os.mkdir(path)
        f1 = path +'/Qeval.pt'
        torch.save(agent.Q_eval.state_dict(),f1)
        f2 = path+'/Q_next.pt'
        torch.save(agent.Q_eval.state_dict(),f2)
        with open(path+'/experiencedataset.npy','wb') as f:
            np.save(f,np.asarray(agent.actionMemory))
        with open(path+'/agent.txt','wb') as f:
            pickle.dump(agent,f)
        print(e)
    except KeyboardInterrupt:
        pwd = os.getcwd()
        filename = f"exploration2-{exploration}"
        path = os.path.join(pwd,filename)
        os.mkdir(path)
        f1 = path +'/Qeval.pt'
        torch.save(agent.Q_eval.state_dict(),f1)
        f2 = path+'/Q_next.pt'
        torch.save(agent.Q_eval.state_dict(),f2)
        with open(path+'/experiencedataset.npy','wb') as f:
            np.save(f,np.asarray(agent.actionMemory))
        with open(path+'/agent.txt','wb') as f:
            pickle.dump(agent,f)
