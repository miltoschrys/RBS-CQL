import subprocess
import sys
import time
host='xxx.xxx.xxx.xxx'
port='xxxxx'
username = 'k8ssandra-superuser'
password ='xxxxxxxxxxxxxxxxxx'
commandfile = "xxxxxxx.cql"

#process1= subprocess.Popen("""ssh -t -t user@83.212.74.253 /home/user/cqlsh-5.1.20/bin/cqlsh {0} {1} -u {2} -p {3} -f /home/user/cqlsh-5.1.20/{4} """.format(host,port,username,password,commandfile)
#                            ,shell=True)
#print(process1.args)
#process1.communicate()

#load ycsb
records=sys.argv[1]
#nodes= sys.argv[2]
target= str(200)
threads=str(4)

ycsbparamfile= '/xxxxxx/xxxxxx/xxxxx/xxxxxxxx.dat'
process1=subprocess.Popen(['ssh','-t','-t','user@client','/home/user/ycsb-0.17.0/bin/ycsb','load','cassandra-cql',
                          '-P','/home/user/ycsb-0.17.0/workloads/workloada','-s','-p','insertcount=500000','-p','insertstart=4500000',
                          '-p','recordcount={0}'.format(records),'-threads',threads,'-target',target,'-P',ycsbparamfile],
                          stderr=subprocess.PIPE,
                          universal_newlines=True)
time.sleep(3)
process2=subprocess.Popen(['ssh','-t','-t','user@client1','/home/user/ycsb-0.17.0/bin/ycsb','load','cassandra-cql',
                          '-P','/home/user/ycsb-0.17.0/workloads/workloada','-s','-p','insertcount=500000','-p','insertstart=5000000',
                          '-p','recordcount={0}'.format(records),'-threads',threads,'-target',target,'-P',ycsbparamfile],
                          stderr=subprocess.PIPE,
                          universal_newlines=True)
time.sleep(3)
process3=subprocess.Popen(['ssh','-t','-t','user@client2','/home/user/ycsb-0.17.0/bin/ycsb','load','cassandra-cql',
                          '-P','/home/user/ycsb-0.17.0/workloads/workloada','-s','-p','insertcount=500000','-p','insertstart=5500000',
                          '-p','recordcount={0}'.format(records),'-threads',threads,'-target',target,'-P',ycsbparamfile],
                          stderr=subprocess.PIPE,
                          universal_newlines=True)
"""
time.sleep(3)
process4=subprocess.Popen(['ssh','-t','-t','user@client3','/home/user/ycsb-0.17.0/bin/ycsb','load','cassandra-cql',
                          '-P','/home/user/ycsb-0.17.0/workloads/workloada','-s','-p','insertcount=1500000','-p','insertstart=4500000',
                          '-p','recordcount={0}'.format(records),'-threads',threads,'-target',target,'-P',ycsbparamfile],
                          stderr=subprocess.PIPE,
                          universal_newlines=True)
"""
p=[process1,process2,process3]
for pr in p:
    pr.communicate()
while True:
    output = process.stderr.readline()
    print(output.strip())
    # Do something else
    return_code = process.poll()
    if return_code is not None:
        print('RETURN CODE', return_code)
        # Process has finished, read rest of the output
        for output in process.stderr.readlines():
            print(output.strip())
        break
