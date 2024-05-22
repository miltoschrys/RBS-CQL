import subprocess
import sys
host='83.212.74.253'
port='30007'
username = 'k8ssandra-superuser'
password ='syLrGYamn6eJqmoonBE6'
commandfile = "cleardb.cql"
#run ycsb

records=sys.argv[1]
operations=sys.argv[2]
target= sys.argv[3]
threads = str(max(int(target)//100,10))
ycsbparamfile= '/home/user/ycsb-0.17.0/ycsbparams.dat'
process=subprocess.Popen(['ssh','-t','-t','user@83.212.74.253','/home/user/ycsb-0.17.0/bin/ycsb','run','cassandra-cql',
                          '-P','/home/user/ycsb-0.17.0/workloads/workloada','-s',
                          '-p','recordcount={0}'.format(records),'-p','operationcount={0}'.format(operations),'-threads',threads,'-target',target,'-P',ycsbparamfile],
                          stderr=subprocess.PIPE,
                          universal_newlines=True)
print(process.args)
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
