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
## add option to initiate with pretrained network
if __name__ =='__main__':

    try:
        """
        f =open('./InitialReplayMemory2.npy','rb')
        ReplayMemory = np.load(f,allow_pickle=True).tolist()
        f.close()
        agent = Agent(gamma=0.99,epsilon=1,alpha=0.001,maxMemorySize=300,
         initialSumR=1866.0404199095283,initialSumSqR=14381.418545208178,epsEnd=0.1,replace_target_cnt=60,
         actionSpace=[0,1,2],init_bias=622.0134733031756,memCntr=len(ReplayMemory),actionMemory=ReplayMemory)
        """
        with open('./training2_463/Qeval.pt','rb') as f2,open('./training2_463/agent.txt','rb') as f3,open('./training2_463/Q_next.pt','rb') as f4:
            Qeval = DeepQNetwork(0.001,0)
            Qeval.load_state_dict(torch.load(f2))
            ##################
            Qnext = DeepQNetwork(0.001,0)
            Qnext.load_state_dict(torch.load(f4))
            ##################
            agent = pickle.load(f3)
            agent.Q_eval = Qeval
            agent.Q_next = Qnext



        env = cluster(8,5,5)
        paramLoader= ParameterLoader(env.get_cluster_size(),minstate=5,maxstate=8)
        MemoryDataset=[]
        scores=[]
        score=0
        epsHist= []
        sigmaHist = []
        batch_size=32
        agent.learn(batch_size)
        if(paramLoader.get_parameters(env.get_cluster_size(),1,'s') ==1):
        #paramLoader.transform_parameters()
            print("error")
            exit()
        reward_dict=deepcopy(paramLoader.reward_dict())
        paramLoader.normalize_parameters()
        observation=paramLoader.parameters_to_nparray()
        #steps=1
        steps= agent.learn_step_counter
        reward=0


        while(steps <=10000):
            print("step: "+str(steps))
            epsHist.append(agent.EPSILON)
            sigmaHist.append(agent.sigmaSq)
            print("choosing action")
            action = agent.chooseAction(observation)
            for (i,value) in zip(range(3),agent.Q_eval.forward(observation)):
                print(f"Qvalue for action {i} is {value}")
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
            reward= 0.01*reward_dict['throughput']  - (reward_dict['vms']-5.0)

            score+= reward
            obs=agent.storeTransition(observation,action,reward,observation_)
            if obs is not None:
                MemoryDataset.append(obs) ## change function to return old transition
            observation_=observation
            agent.learn(batch_size)
            scores.append(reward)
            print(reward)
            steps+=1
            if(steps%100==0 ):
                pwd = os.getcwd()
                filename = f"training2_{steps}"
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



                #print(epshist)
                #print(scores)
                x= np.arange(0,float(len(epsHist)),1.0)
                fig,(ax1,ax3) = plt.subplots(2,1)
                color= 'r'
                ax1.set_xlabel('steps')
                ax1.set_ylabel('epsilon',color=color)
                ax1.plot(x,epsHist,color=color)
                ax1.tick_params(axis='y',labelcolor=color)

                ax2=ax1.twinx()

                color= 'g'

                ax2.set_ylabel('score',color=color)
                ax2.plot(x,scores,color=color)
                ax2.tick_params(axis='y',labelcolor=color)
                ax3.set_xlabel('steps')
                ax3.set_ylabel('loss',color='b')
                x= np.arange(0,float(len(agent.loss)),1.0)
                ax3.plot(x,agent.loss,color='b')
                ax3.tick_params(axis='y',color='b')
                f3 = path + f'/plotScoreEpsilonLoss.png'
                plt.savefig(f3)


    except Exception as e:
        pwd = os.getcwd()
        filename = f"training2_{steps}"
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
        print(e)
    except KeyboardInterrupt:
        pwd = os.getcwd()
        filename = f"training2_{steps}"
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
