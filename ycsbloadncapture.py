import subprocess
host='xxx.xxx.xxx.xxx'
port='xxxxx'
username = 'k8ssandra-superuser'
password ='xxxxxxxxxxxxxxxx'
commandfile = "xxxxx.cql"
process1= subprocess.Popen("""ssh user@xxx.xxx.xxx.xxx /home/user/cqlsh-5.1.20/bin/cqlsh {0} {1} -u {2} -p {3} -f /home/user/cqlsh-5.1.20/{4} """.format(host,port,username,password,commandfile)
                            ,shell=True)
print(process1.args)
process1.communicate()
#load ycsb
threads='10'
target='100'
records='10000'
ycsbparamfile= '/xxx/xxxx/xxxxx/xxxxxx.dat'
process=subprocess.Popen(['ssh','user@xxx.xxx.xxx.xxx','/home/user/ycsb-0.17.0/bin/ycsb','load','cassandra-cql',
                          '-P','/home/user/ycsb-0.17.0/workloads/workloada','-s',
                          '-p','recordcount={0}'.format(records),'-threads',threads,'-target',target,'-P',ycsbparamfile],
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
#run ycsb
operations='30000'
process=subprocess.Popen(['ssh','user@xxx.xxx.xxx.xxx','/home/user/ycsb-0.17.0/bin/ycsb','run','cassandra-cql',
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
#pull prom metrics and write csv file
