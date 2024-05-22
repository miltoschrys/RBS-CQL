import datetime as dt
from prometheus_api_client import PrometheusConnect, MetricRangeDataFrame
import time
import numpy as np
from copy import deepcopy
from math import isnan

class ParameterLoader:
    def __init__(self,state,minstate=5.0,maxstate=8.0,url='http://83.212.74.253:30007'):
        self.prom = PrometheusConnect(url=url)
        self.state=state
        self.maxstate=maxstate
        self.minstate=minstate
        self.maxlatency=1.0
        self.maxthroughput=1500.0
        self.maxmem=self.maxstate*4.0
        self.minmem=self.minstate*4.0
        self.cluster_throughput=0.0

    def get_parameters_from_prometheus(self,state):
        """
        vm_num_json= self.prom.custom_query(query='sum by(cluster)(max by (cluster,datacenter,rack,instance) (changes (mcac_thread_pools_completed_tasks{cluster="k8ssandra",dc="dc1",pool_name="gossip_stage"}[2m:30s])) > bool 0)')
        self.vm_num= float(vm_num_json[0]['value'][1])
        """
        self.vm_num= state
        self.state = state
    #    cluster_latency_json=self.prom.custom_query(query='stargate_client_request_latency_quantile{cluster=~"k8ssandra", dc=~".*", rack=~".*", instance=~".*", quantile=~"0.99", request_type="write"}')
####################
        #self.cluster_latency=float(cluster_latency_json[0]['value'][1])
        cluster_latency_checker = self.prom.custom_query(query='min(stargate_client_request_latency_quantile{cluster=~"k8ssandra", dc=~".*", rack=~".*", instance=~".*", quantile=~"0.999", request_type="read"})')
        lat_checker = float(cluster_latency_checker[0]['value'][1])
        if lat_checker < 0.1 :
            print("one or more stargate nodes are not exposing correct latency")
            cluster_latency_json=self.prom.custom_query(query='max(stargate_client_request_latency_quantile{cluster=~"k8ssandra", dc=~".*", rack=~".*", instance=~".*", quantile=~"0.999", request_type="read"})')
            self.cluster_latency999=float(cluster_latency_json[0]['value'][1])
            cluster_latency_json=self.prom.custom_query(query='max(stargate_client_request_latency_quantile{cluster=~"k8ssandra", dc=~".*", rack=~".*", instance=~".*", quantile=~"0.99", request_type="read"})')
            self.cluster_latency99=float(cluster_latency_json[0]['value'][1])
            cluster_latency_json=self.prom.custom_query(query='max(stargate_client_request_latency_quantile{cluster=~"k8ssandra", dc=~".*", rack=~".*", instance=~".*", quantile=~"0.98", request_type="read"})')
            self.cluster_latency98=float(cluster_latency_json[0]['value'][1])
        else:
            print("normal latency")
            cluster_latency_json=self.prom.custom_query(query='avg(stargate_client_request_latency_quantile{cluster=~"k8ssandra", dc=~".*", rack=~".*", instance=~".*", quantile=~"0.999", request_type="read"})')
            self.cluster_latency999=float(cluster_latency_json[0]['value'][1])
            cluster_latency_json=self.prom.custom_query(query='avg(stargate_client_request_latency_quantile{cluster=~"k8ssandra", dc=~".*", rack=~".*", instance=~".*", quantile=~"0.99", request_type="read"})')
            self.cluster_latency99=float(cluster_latency_json[0]['value'][1])
            cluster_latency_json=self.prom.custom_query(query='avg(stargate_client_request_latency_quantile{cluster=~"k8ssandra", dc=~".*", rack=~".*", instance=~".*", quantile=~"0.98", request_type="read"})')
            self.cluster_latency98=float(cluster_latency_json[0]['value'][1])
        
        self.cluster_throughput_prev=self.cluster_throughput

        cluster_throughput=self.prom.custom_query(query='sum(rate(cql_org_apache_cassandra_metrics_Client_RequestsProcessed_total[5m]))')
        self.cluster_throughput=float(cluster_throughput[0]['value'][1])
        #print(cluster_throughput)
        cluster_total_memory=self.prom.custom_query(query='sum(collectd_memory)')

        self.cluster_total_memory=float(cluster_total_memory[0]['value'][1])

        cluster_cached_memory=self.prom.custom_query(query='sum(collectd_memory{memory="cached"})')
        self.cluster_cached_memory=float(cluster_cached_memory[0]['value'][1])

        cluster_free_memory=self.prom.custom_query(query='sum(collectd_memory{memory="free"})')

        self.cluster_free_memory=float(cluster_free_memory[0]['value'][1])

        cluster_buffered_memory=self.prom.custom_query(query='sum(collectd_memory{memory="buffered"})')

        self.cluster_buffered_memory=float(cluster_buffered_memory[0]['value'][1])

        cluster_avg_total_cpu=self.prom.custom_query(query='avg by (cluster)( sum by (cluster, dc, rack, instance) (rate(collectd_cpu_total{cluster="k8ssandra", dc=~".*", rack=~".*", instance=~".*"}[1m:30s])) )')

        self.cluster_avg_total_cpu=float(cluster_avg_total_cpu[0]['value'][1])
        while(isnan(self.cluster_avg_total_cpu)):
            print("waiting for cpu metric")
            time.wait(10)
            cluster_avg_total_cpu=self.prom.custom_query(query='avg by (cluster)( sum by (cluster, dc, rack, instance) (rate(collectd_cpu_total{cluster="k8ssandra", dc=~".*", rack=~".*", instance=~".*"}[1m:30s])) )')

            self.cluster_avg_total_cpu=float(cluster_avg_total_cpu[0]['value'][1])


        cluster_avg_idle_cpu=self.prom.custom_query(query='avg by (cluster) (sum by (cluster, dc, rack, instance) (rate(collectd_cpu_total{type="idle", cluster="k8ssandra", dc=~".*", rack=~".*", instance=~".*"}[1m:30s])) )')

        self.cluster_avg_idle_cpu=float(cluster_avg_idle_cpu[0]['value'][1])

        cluster_avg_busywait_cpu=self.prom.custom_query(query='avg by (cluster) (sum by (cluster, dc, rack, instance) (rate(collectd_cpu_total{type="wait", cluster="k8ssandra", dc=~".*", rack=~".*", instance=~".*"}[1m:30s])) )')

        self.cluster_avg_busywait_cpu=float(cluster_avg_busywait_cpu[0]['value'][1])

        max_cluster_cpu_used= self.prom.custom_query(query='max by (cluster) (1 - (sum by (cluster, dc, rack, instance) (rate(collectd_cpu_total{type="idle", cluster="k8ssandra", dc=~".*", rack=~".*", instance=~".*"}[1m:30s])) / sum by (cluster, dc, rack, instance) (rate(collectd_cpu_total{cluster="k8ssandra", dc=~".*", rack=~".*", instance=~".*"}[1m:30s]))))')

        min_cluster_cpu_used= self.prom.custom_query(query='min by (cluster) (1 - (sum by (cluster, dc, rack, instance) (rate(collectd_cpu_total{type="idle", cluster="k8ssandra", dc=~".*", rack=~".*", instance=~".*"}[1m:30s])) / sum by (cluster, dc, rack, instance) (rate(collectd_cpu_total{cluster="k8ssandra", dc=~".*", rack=~".*", instance=~".*"}[1m:30s]))))')

        self.max_cluster_cpu_used=float(max_cluster_cpu_used[0]['value'][1])

        self.min_cluster_cpu_used=float(min_cluster_cpu_used[0]['value'][1])

        cluster_avg_read_iops=self.prom.custom_query(query=r'avg by (cluster) (sum by (cluster, dc, rack, instance) (irate(collectd_disk_disk_ops_read_total{cluster="k8ssandra",instance=~".*", disk=~".*\\d+"}[5m])))')

        self.cluster_avg_read_iops=float(cluster_avg_read_iops[0]['value'][1])

        cluster_avg_write_iops=self.prom.custom_query(query=r'avg by (cluster) (sum by (cluster, dc, rack, instance) (irate(collectd_disk_disk_ops_write_total{cluster="k8ssandra",instance=~".*", disk=~".*\\d+"}[5m])))')

        self.cluster_avg_write_iops=float(cluster_avg_write_iops[0]['value'][1])

    def get_parameters(self,state,action,window):
        try:
            """
            vms_json=self.prom.custom_query(query='sum by(cluster)(max by (cluster,datacenter,rack,instance) (changes (mcac_thread_pools_completed_tasks{cluster="k8ssandra",dc="dc1",pool_name="gossip_stage"}[2m:30s])) > bool 0)')
            vms= int(vms_json[0]['value'][1])
            """
            vms = state
            """
            count = 30
            while (vms != state):
                print("vms not equal to expected state. waiting..\n")
                time.sleep(30)
                count-=1

                ##if count <0:
                ##    print("state of the cluster does not match with expected state")
                ##    return [-1]
                vms_json=self.prom.custom_query(query='sum by(cluster)(max by (cluster,datacenter,rack,instance) (changes (mcac_thread_pools_completed_tasks{cluster="k8ssandra",dc="dc1",pool_name="gossip_stage"}[2m:30s])) > bool 0)')
                vms= int(vms_json[0]['value'][1])
            """
            if(window=='s'):
                print("small wait window")
                print("waiting 2:30min for new action to take effect")
                time.sleep(150)
            else:
                print("waiting 5 min for new action to take effect")
                time.sleep(300)
            self.get_parameters_from_prometheus(vms)
            return 0
        except :
            print("Connection problem with PromClient")
            return 1

    def transform_parameters(self):
        self.vm_num = float(self.vm_num[0]['value'][1])
        self.cluster_latency=float(self.cluster_latency[0]['value'][1])
        self.cluster_throughput=float(self.cluster_throughput[0]['value'][1])
        self.cluster_total_memory=float(self.cluster_total_memory[0]['value'][1])
        self.cluster_free_memory=float(self.cluster_free_memory[0]['value'][1])
        self.cluster_avg_idle_cpu=float(self.cluster_avg_idle_cpu[0]['value'][1])
        self.cluster_avg_read_iops=float(self.cluster_avg_read_iops[0]['value'][1])
        self.cluster_avg_total_cpu=float(self.cluster_avg_total_cpu[0]['value'][1])
        self.cluster_cached_memory=float(self.cluster_cached_memory[0]['value'][1])
        self.cluster_buffered_memory=float(self.cluster_buffered_memory[0]['value'][1])
        self.cluster_avg_busywait_cpu=float(self.cluster_avg_busywait_cpu[0]['value'][1])
        self.cluster_avg_write_iops=float(self.cluster_avg_write_iops[0]['value'][1])

    def normalize_parameters(self):
        self.vm_num_n=(self.vm_num-self.minstate)/ (self.maxstate - self.minstate)
        self.cluster_latency999_n = self.cluster_latency999/self.maxlatency
        self.cluster_latency99_n = self.cluster_latency99/self.maxlatency
        self.cluster_latency98_n = self.cluster_latency98/self.maxlatency
        self.cluster_throughput_n=self.cluster_throughput/self.maxthroughput
        self.cluster_free_memory=self.cluster_free_memory/self.cluster_total_memory
        self.cluster_cached_memory=self.cluster_cached_memory/self.cluster_total_memory
        self.cluster_buffered_memory=self.cluster_buffered_memory/self.cluster_total_memory
        #maybe change total mem
        self.cluster_total_memory=(self.state*4.0-self.minmem)/(self.maxmem-self.minmem)
        self.cluster_avg_idle_cpu=self.cluster_avg_idle_cpu/self.cluster_avg_total_cpu
        self.cluster_avg_busywait_cpu=self.cluster_avg_busywait_cpu/self.cluster_avg_total_cpu
        self.cluster_avg_read_iops=self.cluster_avg_read_iops/1500.0
        self.cluster_avg_write_iops=self.cluster_avg_write_iops/1500.0

    def parameters_to_nparray(self):
        parameterlist=[self.vm_num_n,self.cluster_latency999_n,self.cluster_latency99_n,self.cluster_latency98_n,
                       self.cluster_throughput_n,self.cluster_free_memory,self.cluster_cached_memory,self.cluster_buffered_memory,
                       self.cluster_avg_idle_cpu,self.cluster_avg_busywait_cpu,self.max_cluster_cpu_used,
                       self.min_cluster_cpu_used,self.cluster_avg_read_iops,self.cluster_avg_write_iops,self.cluster_throughput_prev]
        return np.array(parameterlist)

    def reward_dict(self):
        vms = self.vm_num
        latency= self.cluster_latency98
        throughput = self.cluster_throughput
        return {'vms': vms ,'latency':latency ,'throughput':throughput}






if __name__ == '__main__':
    loader=ParameterLoader(8)
    loader.get_parameters_from_prometheus(8)
    print(loader.vm_num)
    print("cluster_latency999: ")

    print(loader.cluster_latency999)
    print("cluster_latency99: ")

    print(loader.cluster_latency99)
    print("cluster_latency98: ")

    print(loader.cluster_latency98)
    print("\n")
    print("cluster_throughput: ")
    print(loader.cluster_throughput)
    print("\n")
    print("total_mem:")
    print(loader.cluster_total_memory)
    print("\n")
    print("free mem:")
    print(loader.cluster_free_memory)
    print("\n")
    print("avg idle cpu: ")
    print(loader.cluster_avg_idle_cpu)
    print("\n")
    print("avg_read_iops")
    print(loader.cluster_avg_read_iops)
    print("\n")
    print("avg total cp")
    print(loader.cluster_avg_total_cpu)
    print("\n")
    print("cached mem : ")
    print(loader.cluster_cached_memory)
    print("\n")
    print("buffered mem")
    print(loader.cluster_buffered_memory)
    print("\n")
    print("busywait cpu")
    print(loader.cluster_avg_busywait_cpu)
    print("\n")
    print("write iops")
    print(loader.cluster_avg_write_iops)
    print("\n")
    print("max_cpu_used")
    print(loader.max_cluster_cpu_used)
    print("\n")
    print("min_cpu_used")
    print(loader.min_cluster_cpu_used)






#    loader.transform_parameters()
    print(loader.reward_dict())
    loader.normalize_parameters()
    print(loader.parameters_to_nparray())
    print(loader.reward_dict())
