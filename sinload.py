import subprocess
import sys
import time
import random
import os
records=sys.argv[1]
operations= str(48*60*60*50)
threads = "2"
target = "50"

ycsbparamfile= '/home/user/ycsb-0.17.0/ycsbparams.dat'
clients = ['user@client2','user@client','user@client1','user@client3']
proc=[]
for i in range(0,10):
    time.sleep(10)
    proc.append(subprocess.Popen(['ssh','-t','-t',clients[i%4],'/home/user/ycsb-0.17.0/bin/ycsb','run','cassandra-cql',
                          '-P','/home/user/ycsb-0.17.0/workloads/workloada','-s',
                          '-p','recordcount={0}'.format(records),'-p','operationcount={0}'.format(operations),'-threads',threads,'-target',target,'-P',ycsbparamfile]
                          )
                )

m=5
loads = ['/home/user/ycsb-0.17.0/workloads/workloada','/home/user/ycsb-0.17.0/workloads/workloadb']
for j in range(0,18):
    proc = proc[0:20] #############
    for i in range(0,20):
        """
        if j%3 ==0:
            load = 1
        if j%3 == 1:
            load = random.randint(0,1)
        if j%3 == 2:
        """
        load = 0
        print("starting "+str(i))
        print(f"load should be about {700+50*(i+1)}")
        ##operations= str(m*50*60+2*m*60*50*(25-i))
        executiontime=str((m*60 + 2*m*60*(300-i)))
        proc.append(subprocess.Popen(['ssh','-t','-t',clients[i%4],'/home/user/ycsb-0.17.0/bin/ycsb','run','cassandra-cql',
                                  '-P',loads[load],'-s','-p','maxexecutiontime={0}'.format(executiontime),
                                  '-p','recordcount={0}'.format(records),'-p','operationcount={0}'.format(operations),'-threads',threads,'-target',target,'-P',ycsbparamfile]
                                  )
                    )

        time.sleep(60*m)
    proc[20].communicate()
##    proc[4].communicate()

    time.sleep(60*m)
"""
    for i in range(0,20):

        if j%3 ==0:
            load = 1
        if j%3 == 1:
            load = random.randint(0,1)
        if j%3 == 2:

        load = random.randint(0,1)
        print("starting "+str(i))
        time.sleep(60*m)
        operations= str(15*50*60+2*m*60*50*(19-i))
        proc.append(subprocess.Popen(['ssh','-t','-t',clients[i%2],'/home/user/ycsb-0.17.0/bin/ycsb','run','cassandra-cql',
                                  '-P',loads[load],'-s',
                                  '-p','recordcount={0}'.format(records),'-p','operationcount={0}'.format(operations),'-threads',threads,'-target',target,'-P',ycsbparamfile],
                                   stderr=subprocess.PIPE,
                                  universal_newlines=True)
                    )
        time.sleep(60*m)
    proc[0].communicate()
    time.sleep(60*m)
    proc=[]
    for i in range(0,18):

        if j%3 ==0:
            load = 1
        if j%3 == 1:
            load = random.randint(0,1)
        if j%3 == 2:

        load = 1
        print("starting "+str(i))
        time.sleep(60*m)
        operations= str(15*50*60+2*m*60*50*(17-i))
        proc.append(subprocess.Popen(['ssh','-t','-t',clients[i%2],'/home/user/ycsb-0.17.0/bin/ycsb','run','cassandra-cql',
                                  '-P',loads[load],'-s',
                                  '-p','recordcount={0}'.format(records),'-p','operationcount={0}'.format(operations),'-threads',threads,'-target',target,'-P',ycsbparamfile],
                                   stderr=subprocess.PIPE,
                                  universal_newlines=True)
                    )
        time.sleep(60*m)
    for p in proc:
        p.wait()
    proc[0].communicate()
    time.sleep(60*m)
"""
proc[0].communicate()
sys.exit(0)
