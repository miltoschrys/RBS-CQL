import subprocess
import sys

nodes=sys.argv[1]
load=sys.argv[2]
records=str(int(load) * int(nodes))
print(nodes+" nodes 10% load loading \n")
process=subprocess.Popen(['python3','ycsbload.py',records,nodes],
                          stdout=subprocess.PIPE,
                          universal_newlines=True)
print(process.args)
while True:
    output = process.stdout.readline()
    print(output.strip())
    # Do something else
    return_code = process.poll()
    if return_code is not None:
        print('RETURN CODE', return_code)
        # Process has finished, read rest of the output
        for output in process.stdout.readlines():
            print(output.strip())
        break
throughput=['200','400','800','1000','1200','1500','2000','2200','2500','2700','3000','3200']##'800','1000']
load='10'
for target in throughput:
    print(nodes+" nodes 10%load running "+target+" throughput\n")
    operations=str(int(target)*600)##********
    process=subprocess.Popen(['python3','ycsbrun.py',records,operations,target],
                             stdout=subprocess.PIPE,
                             universal_newlines=True)
    print(process.args)
    while True:
        output = process.stdout.readline()
        print(output.strip())
        # Do something else
        return_code = process.poll()
        if return_code is not None:
            print('RETURN CODE', return_code)
            # Process has finished, read rest of the output

            break
    print("collecting metrics for "+target+" throughput\n")
    process=subprocess.Popen(['python3','promclient.py',load,nodes,target])
    process.communicate()
