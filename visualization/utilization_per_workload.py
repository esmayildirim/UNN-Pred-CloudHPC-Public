import matplotlib.pyplot as plt
import json

def visualize(path, cluster_name):
    
    with open(path, 'r') as f:
        workloads= json.load(f)
    print(len(workloads), " workload, ", len(workloads[0]["tasklist"]), " tasks.")
    for workload in workloads:
        runtimes = []
        taskids = []
        cpus = []
        gpus = []
        start_times = []
        finish_times = []
        for i in range(len(workload["tasklist"])):
            task = workload["tasklist"][i]
            taskids.append(int(task["task_id"]))
            runtimes.append(float(task["finish_time"]) - float(task["start_time"]))
            #if (float(task["finish_time"]) - float(task["start_time"]))> 1700000000:
            #    print(task["finish_time"]) - float(task["start_time"])
            start_times.append(float(task["start_time"]))
            finish_times.append(float(task["finish_time"]))
            cpus.append(int(task["cpus"]))
            gpus.append(int(float(task["gpus"])))
        elapsed = (max(finish_times) - min(start_times))/60
        print(elapsed, "minutes.")
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(cluster_name + ' Cluster', fontsize=12)
        barWidth = 0.25
        # creating the bar plot
        p1 = axs[0].bar(taskids, cpus, color ='blue', width = barWidth)
        taskids2 = [x + barWidth for x in taskids]
        p2 = axs[0].bar(taskids2, gpus, color ='green', width = barWidth)
        axs[0].legend((p1[0], p2[0]), ('cpus', 'gpus'))
        axs[0].set_xlabel("Tasks")
        axs[0].set_ylabel("#")
        axs[0].set_xticks(taskids)
        axs[0].set_xticklabels(taskids, fontsize=10, rotation=45)
        axs[0].set_title("# cpus/gpus per task")
        p3 = axs[1].bar(taskids, runtimes, color ='red', width = barWidth)
        #axs[0].legend((p1[0]), (''))
        axs[1].set_xlabel("Tasks")
        axs[1].set_ylabel("Secs")
        axs[1].set_xticks(taskids)
        axs[1].set_xticklabels(taskids, fontsize=10)
        axs[1].set_title("Runtime per task")
        plt.show()
    #plt.savefig(cluster_name+"_task_stats.png", dpi=300)
def sub_min(list):
    if len(list)>0:
        minimum = min(list)
        for i in range(len(list)):
            list[i] -= minimum
    return list

def visualize_util(path,cluster_name):

    with open(path, 'r') as f:
        workloads= json.load(f)
    
    
    count = 0
    for workload in workloads:
        #print(len(workloads), " workload, ", len(workload["node_list"]), " nodes.")
        fig, axs = plt.subplots(4, len(workload["node_list"]), figsize=(16, 10))
        fig.suptitle(cluster_name + ' Cluster - Mixed Workload', fontsize=12)
        #fig.tight_layout()
        for i in range(len(workload["node_list"])):
            time_cpu = []
            cpu_util = []
            time_gpu = []
            #gpu_util = []
            gpu_utils = [[],[],[],[]]
            time_network = []
            network_in = []
            network_out = []
            time_disk = []
            disk_read = []
            disk_write = []
            time_memory = []
            memory_util = []
            node = workload["node_list"][i]
            util_list = node["metrics"]["cpu_util"]
            if len(util_list)>0:
                for j in range(len(util_list)):
                    time_cpu.append(float(util_list[j][0]))
                    cpu_util.append(float(util_list[j][1]))
            util_list = node["metrics"]["memory_util"]
            if len(util_list)>0:
                for j in range(len(util_list)):
                    time_memory.append(float(util_list[j][0]))
                    memory_util.append(float(util_list[j][1]))
        
            util_list = node["metrics"]["gpu_util"]
            if len(util_list)>0:
                print("Hello "+ cluster_name)
                util_list = sorted(util_list,key=lambda l:l[0])
                gpu_utils = [[],[],[],[]]
                time_gpu = []
                for j in range(len(util_list)):
                    time_gpu.append(float(util_list[j][0]))
                    for k in range(1, 5):
                        if(k<len(util_list[j])):
                            gpu_utils[k-1].append(float(util_list[j][k]))
                        else: gpu_utils[k-1].append(0.0)
                        
            util_list = node["metrics"]["network_util"]
            if len(util_list)>0:
                for j in range(len(util_list)):
                    if float(util_list[j][1]) > 1e9:
                        print("in", float(util_list[j][0]))
                    if float(util_list[j][2])> 1e9:
                        print("out",float(util_list[j][0]))
                    time_network.append(float(util_list[j][0]))
                    network_in.append(float(util_list[j][1]))
                    network_out.append(float(util_list[j][2]))
            util_list = node["metrics"]["disk_util"]
            if len(util_list)>0:
                for j in range(len(util_list)):
                    time_disk.append(float(util_list[j][0]))
                    disk_read.append(float(util_list[j][1]))
                    disk_write.append(float(util_list[j][2]))
            
            time_cpu = sub_min(time_cpu)
            time_gpu = sub_min(time_gpu)
            time_network = sub_min(time_network)
            time_disk = sub_min(time_disk)
            time_memory = sub_min(time_memory)
            #markers=['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']
            p1 = axs[0,i].plot(time_cpu, cpu_util,label = 'cpu_util')
            for k in range(len(gpu_utils)):
                if len(gpu_utils[k]) > 0:
                    p = axs[0,i].plot(time_gpu, gpu_utils[k], label = ("gpu_util "+str(k)))
                    
            #axs[0,i].legend((p1[0], p2[0]), ('cpu_util', 'gpu_util'))
            axs[0,i].legend()
            axs[0,i].set_xlabel("Time in secs")
            axs[0,i].set_ylabel("%")
            axs[0,i].set_title("CPU/GPU Utilization - Node "+ str(i))
            
            p3 = axs[1,i].plot(time_network, network_in)
            p4 = axs[1,i].plot(time_network, network_out)
            axs[1,i].legend((p3[0], p4[0]), ('network_in', 'network_out'))
            axs[1,i].set_xlabel("Time in secs")
            axs[1,i].set_ylabel("Bytes")
            axs[1,i].set_title("Network Utilization - Node "+ str(i))

            p5 = axs[2,i].plot(time_disk, disk_read)
            p6 = axs[2,i].plot(time_disk, disk_write)
            axs[2,i].legend((p5[0], p6[0]), ('disk_read', 'disk_write'))
            axs[2,i].set_xlabel("Time in secs")
            axs[2,i].set_ylabel("Bytes")
            axs[2,i].set_title("Disk Utilization - Node "+ str(i))
            
            p7 = axs[3,i].plot(time_memory, memory_util,label = 'memory_util')
            
            #axs[0,i].legend((p1[0], p2[0]), ('cpu_util', 'gpu_util'))
            axs[3,i].legend()
            axs[3,i].set_xlabel("Time in secs")
            axs[3,i].set_ylabel("%")
            axs[3,i].set_title("Memory Utilization - Node "+ str(i))
        plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.30,
                        hspace=0.45)
        #plt.show()
        plt.savefig(cluster_name+"_system_load"+str(count)+".png", dpi=300)
        count += 1
if __name__ == "__main__":

    visualize("../data/ic2/workloads_mixed.json", "IC2")
    #visualize("../data/polaris/workloads_mixed.json","Polaris")
    #visualize("../data/aws/workloads_scattered.json","AWS")
    
    visualize_util("../data/ic2/system_load_mixed.json", "IC2")
    #visualize_util("../data/polaris/system_load_mixed.json", "Polaris")
    #visualize_util("../data/aws/system_load_scattered.json", "AWS")
