import json

def calc_node_cpu_util(nodelist, start_time, end_time):
    ave_cpu_util = 0.0
    for node in nodelist:
        ave_cpu_util_per_node = 0.0
        cpu_utils = node["metrics"]["cpu_util"]
        count = 0
        for util in cpu_utils:
            if float(util[0]) >= start_time and float(util[0]) <= end_time:
                ave_cpu_util_per_node += float(util[1])
                count += 1
        if count > 0:
            ave_cpu_util_per_node /= count
        ave_cpu_util += ave_cpu_util_per_node
    ave_cpu_util /= len(nodelist)
    return ave_cpu_util

def calc_node_memory_util(nodelist, start_time, end_time):
    ave_memory_util = 0.0
    for node in nodelist:
        ave_memory_util_per_node = 0.0
        memory_utils = node["metrics"]["memory_util"]
        count = 0
        for util in memory_utils:
            if float(util[0]) >= start_time and float(util[0]) <= end_time:
                ave_memory_util_per_node += float(util[1])
                count += 1
        if count > 0:
            ave_memory_util_per_node /= count
        ave_memory_util += ave_memory_util_per_node
    ave_memory_util /= len(nodelist)
    return ave_memory_util
    
def calc_node_gpu_util(nodelist, start_time, end_time):
    ave_gpu_util = 0.0
    for node in nodelist:
        ave_gpu_util_per_node = 0.0
        gpu_utils = node["metrics"]["gpu_util"]
        if len(gpu_utils)>0:
            count = 0
            for util in gpu_utils:
                if float(util[0]) >= start_time and float(util[0]) <= end_time:
                    for k in range(1, len(util)):
                        ave_gpu_util_per_node += float(util[k])
                        count += 1
            if count>0:
                ave_gpu_util_per_node /= count
            ave_gpu_util += ave_gpu_util_per_node
    ave_gpu_util /= len(nodelist)
    return ave_gpu_util
    
def calc_node_network_util(nodelist, start_time, end_time, metric_index): # 1 for in 2 for out
    ave_net_util = 0.0
    for node in nodelist:
        ave_net_util_per_node = 0.0
        net_utils = node["metrics"]["network_util"]
        count = 0
        for util in net_utils:
            if float(util[0]) >= start_time and float(util[0]) <= end_time:
                ave_net_util_per_node += float(util[metric_index])
                count += 1
        if count > 0:
            ave_net_util_per_node /= count
        ave_net_util += (ave_net_util_per_node )
    ave_net_util /= (len(nodelist) * 1024 *1024 * 5) # 5 second Bs to MBs per second
    ave_net_util *= 8 #Mbps
    return ave_net_util

def calc_node_disk_util(nodelist, start_time, end_time, metric_index): # 1 for in 2 for out
    ave_disk_util = 0.0
    for node in nodelist:
        ave_disk_util_per_node = 0.0
        disk_utils = node["metrics"]["disk_util"]
        count = 0
        for util in disk_utils:
            if float(util[0]) >= start_time and float(util[0]) <= end_time:
                ave_disk_util_per_node += float(util[metric_index])
                count += 1
        if count > 0:
            ave_disk_util_per_node /= count
        ave_disk_util += (ave_disk_util_per_node )
    ave_disk_util /= (len(nodelist) * 1024 * 1024 * 5) # 5 second Bs to MBs per second
    ave_disk_util *= 8 # Mbps
    return ave_disk_util

