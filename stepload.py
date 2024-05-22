import subprocess
import sys
import time
import random
records=sys.argv[1]
operations= str(48*60*60*200)
threads = "2"
target = "200"
ycsbparamfile= '/home/user/ycsb-0.17.0/ycsbparams.dat'
process=subprocess.Popen(['ssh','-t','-t','user@client','/home/user/ycsb-0.17.0/bin/ycsb','run','cassandra-cql',
                          '-P','/home/user/ycsb-0.17.0/workloads/workloadb','-s',
                          '-p','recordcount={0}'.format(records),'-p','operationcount={0}'.format(operations),'-threads',threads,'-target',target,'-P',ycsbparamfile],
                          stderr=subprocess.PIPE,
                          universal_newlines=True)
m=120
clients = ['user@client2','user@client']
loads = ['/home/user/ycsb-0.17.0/workloads/workloada','/home/user/ycsb-0.17.0/workloads/workloadb']
print("part1")
proc=[]
for i in range(0,12):
    print("starting "+str(i))
    time.sleep(120*60)
    operations= str((12-i)*m*60*200)
    proc.append(subprocess.Popen(['ssh','-t','-t',clients[0],'/home/user/ycsb-0.17.0/bin/ycsb','run','cassandra-cql',
                              '-P',loads[random.randint(0,1)],'-s',
                              '-p','recordcount={0}'.format(records),'-p','operationcount={0}'.format(operations),'-threads',threads,'-target',target,'-P',ycsbparamfile],
                               stderr=subprocess.PIPE,
                              universal_newlines=True)
                )
proc[0].communicate()
proc=[]
print("part2")
for i in range(0,8):
    print("starting "+str(i))
    time.sleep(120*60)
    operations= str((8-i)*m*60*200)
    proc.append(subprocess.Popen(['ssh','-t','-t',clients[1],'/home/user/ycsb-0.17.0/bin/ycsb','run','cassandra-cql',
                              '-P',loads[random.randint(0,1)],'-s',
                              '-p','recordcount={0}'.format(records),'-p','operationcount={0}'.format(operations),'-threads',threads,'-target',target,'-P',ycsbparamfile],
                               stderr=subprocess.PIPE,
                              universal_newlines=True)
                )
proc[0].communicate()
proc=[]
print("part3")
for i in range(0,10):
    print("starting "+str(i))
    time.sleep(120*60)
    operations= str((10-i)*m*60*200)
    proc.append(subprocess.Popen(['ssh','-t','-t',clients[i%2],'/home/user/ycsb-0.17.0/bin/ycsb','run','cassandra-cql',
                              '-P',loads[random.randint(0,1)],'-s',
                              '-p','recordcount={0}'.format(records),'-p','operationcount={0}'.format(operations),'-threads',threads,'-target',target,'-P',ycsbparamfile],
                               stderr=subprocess.PIPE,
                              universal_newlines=True)
                )
proc[0].communicate()


process.communicate()
sys.exit(0)
