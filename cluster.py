import paramiko
import time
from constants import CLUSTER_CONF as CONF

class cluster(object):
    def __init__(self,maxstate,minstate,state):
        self.maxstate=maxstate
        self.minstate=minstate
        self.state=state
        self.con= paramiko.SSHClient()
        self.con.load_system_host_keys()
        self.con.connect(CONF['ip'],username=CONF['username'],password=CONF['password'])
    def execute(self,action):
        if action not in [0,1,2]:
            print(f"invalid action {action}. Expected 0,1 or 2\n")
        if action == 1:
            return 's'
        if action == 0:
            return self._decr_cluster(1)
        if action == 2:
            return self._incr_cluster(1)


    def _decr_cluster(self,i):
        self.state-= i
        command= f"helm upgrade k8ssandra k8ssandra/k8ssandra --reuse-values --set cassandra.datacenters\\[0\\].size={self.state},cassandra.datacenters\\[0\\].name=dc1"
        if self.state  < self.minstate:
            self.state=self.minstate
            print("cluster already at min size\n")
            return 's'
        stdin,stdout,stderr=self.con.exec_command(command)
        print(stdout.read().decode())
        err= stderr.read().decode()
        if err:
            print(err)
        return 'l'

    def _incr_cluster(self,i):
        self.state+=i
        command= f"helm upgrade k8ssandra k8ssandra/k8ssandra --reuse-values --set cassandra.datacenters\\[0\\].size={self.state},cassandra.datacenters\\[0\\].name=dc1"
        if self.state > self.maxstate:
            self.state=self.maxstate
            print("cluster already at max size\n")
            return 's'
        stdin,stdout,stderr=self.con.exec_command(command)
        print(stdout.read().decode())
        err= stderr.read().decode()
        if err:
            print(err)
        return 'l'

    def restart_metrics_collector(self):
        command="kubectl delete pod prometheus-k8ssandra-kube-prometheus-prometheus-0"
        stdin,stdout,stderr=self.con.exec_command(command)
        print(stdout.read().decode())
        err= stderr.read().decode()
        if err:
            print(err)
        return

    def check_operator_rediness(self):

            command="kubectl get cassdc/dc1 -o \"jsonpath={.status.cassandraOperatorProgress}\""
            stdin,stdout,stderr= self.con.exec_command(command)
            if(stdout.read().decode()=='Ready'):
                return True
            else:
                return False

    def get_cluster_size(self):
        command="kubectl get cassdc/dc1 -o \"jsonpath={.spec.size}\""
        stdin,stdout,stderr= self.con.exec_command(command)
        self.state = float(stdout.read().decode())
        return self.state
