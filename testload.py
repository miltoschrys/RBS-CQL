import subprocess
import sys
import time
import random
import os

records=sys.argv[1]
operations= str(48*60*60*200)
threads = "2"
target = "50"
ycsbparamfile= '/home/user/ycsb-0.17.0/ycsbparams.dat'
proc=[]
proc.append(subprocess.Popen(['ssh','-t','-t','user@client','/home/user/ycsb-0.17.0/bin/ycsb','run','cassandra-cql',
                          '-P','/home/user/ycsb-0.17.0/workloads/workloadb','-s',
                          '-p','recordcount={0}'.format(records),'-p','operationcount={0}'.format(operations),'-threads',threads,'-target',target,'-P',ycsbparamfile],
                          stderr=subprocess.PIPE,
                          universal_newlines=True,start_new_session=True)
            )
m=10
clients = ['user@client2','user@client','user@client1','user@client3']
loads = ['/home/user/ycsb-0.17.0/workloads/workloadf','/home/user/ycsb-0.17.0/workloads/workloadb','/home/user/ycsb-0.17.0/workloads/workloada']
proc=[]
#random.randint(0,1)
for i in range(0,10):

    time.sleep(10)
    operations= str(m*60*100*(200-i))
    executiontime= str((10-i)*3*60)
    proc.append(subprocess.Popen(['ssh','-t','-t',clients[i%4],'/home/user/ycsb-0.17.0/bin/ycsb','run','cassandra-cql',
                              '-P',loads[2],'-s','-p','maxexecutiontime={0}'.format(executiontime),
                              '-p','recordcount={0}'.format(records),'-p','operationcount={0}'.format(operations),'-threads',threads,'-target',target,'-P',ycsbparamfile],
                               stderr=subprocess.PIPE,start_new_session=True)
                )


"""
for p in proc.reverse():
    try:
        p.wait(2*60)
    except subprocess.TimeoutExpired:
        print("expired wait window")
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
"""
##time.sleep(120*60)
"""
for i in range(0,10):
    time.sleep(m*60)
    operations= str(m*60*50*(10-i))
    proc.append(subprocess.Popen(['ssh','-t','-t',clients[i%2],'/home/user/ycsb-0.17.0/bin/ycsb','run','cassandra-cql',
                              '-P',loads[1],'-s',
                              '-p','recordcount={0}'.format(records),'-p','operationcount={0}'.format(operations),'-threads',threads,'-target',target,'-P',ycsbparamfile],
                               stderr=subprocess.PIPE,
                              universal_newlines=True)
                              )
"""
proc[0].communicate()

sys.exit(0)
