import datetime as dt
from prometheus_api_client import PrometheusConnect, MetricRangeDataFrame
import sys

load=sys.argv[1]
nodes=sys.argv[2]
target=sys.argv[3]

prom=PrometheusConnect(url='http://83.212.74.253:30009')
with open("/Users/chrrys/projects/python3/thesis/prometheus_values.txt","r") as f:
    for i in range(514):
        metric=f.readline().strip()
        print(metric)
        if metric :
            metric_data = prom.get_metric_range_data(metric_name=metric,start_time=(dt.datetime.now()-dt.timedelta(minutes=10)),end_time=dt.datetime.now())
        if metric_data:
            metric_df= MetricRangeDataFrame(metric_data)
            metric_df.to_pickle("dataset/load"+load+"/"+nodes+"/"+target+"/"+metric+".pkl")
            metric_df.to_csv("dataset/load"+load+"/"+nodes+"/"+target+"/"+metric+".csv")
            print("stored\n")
