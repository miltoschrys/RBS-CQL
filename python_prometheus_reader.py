import json
with open("prometheus_values.json","r") as f:
    data=json.load(f)
    f2=open("prometheus_values.txt","w")
    print(data.keys())
    for i in data['data']:
       f2.write(i+"\n")
    f2.close()
f.close()
