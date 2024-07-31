import json

def calc_avg_network_util(task, start_time = None):
    ave_net_in = 0.0
    ave_net_out = 0.0
    for j in range(len(task["nodes"])):
        ave_net_in_per_node = 0.0
        ave_net_out_per_node = 0.0
        count = 0
        for k in range(len(task["nodes"][j]["metrics"]["network_util"])):
            if start_time != None and float(task["nodes"][j]["metrics"]["network_util"][k][0]) >= float(start_time):
                continue
            ave_net_in_per_node += float(task["nodes"][j]["metrics"]["network_util"][k][1])
            ave_net_out_per_node += float(task["nodes"][j]["metrics"]["network_util"][k][2])
            count += 1
        if count > 0:
            ave_net_in_per_node /= count
            ave_net_out_per_node /= count
        ave_net_in += ave_net_in_per_node
        ave_net_out += ave_net_out_per_node
    ave_net_in /= (len(task["nodes"]) * 1024 *1024 * 5)
    ave_net_out /= (len(task["nodes"]) * 1024 * 1024 * 5)
    task["avg_network_in_Mbps"] = ave_net_in * 8
    task["avg_network_out_Mbps"] = ave_net_out * 8

def calc_avg_disk_util(task, start_time = None):
    ave_disk_read = 0.0
    ave_disk_write = 0.0
    for j in range(len(task["nodes"])):
        ave_disk_read_per_node = 0.0
        ave_disk_write_per_node = 0.0
        count = 0
        for k in range(len(task["nodes"][j]["metrics"]["disk_util"])):
            if start_time != None and float(task["nodes"][j]["metrics"]["disk_util"][k][0]) >= float(start_time):
                continue
            ave_disk_read_per_node += float(task["nodes"][j]["metrics"]["disk_util"][k][1])
            ave_disk_write_per_node += float(task["nodes"][j]["metrics"]["disk_util"][k][2])
            count += 1
        if count > 0:
            ave_disk_read_per_node /= count
            ave_disk_write_per_node /= count
        ave_disk_read += ave_disk_read_per_node
        ave_disk_write += ave_disk_write_per_node
    ave_disk_read /= (len(task["nodes"]) * 1024 * 1024* 5)
    ave_disk_write /= (len(task["nodes"]) * 1024 * 1024 *5)
    task["avg_disk_read_Mbps"] = ave_disk_read * 8
    task["avg_disk_write_Mbps"] = ave_disk_write * 8
def calc_avg_cpu_util(task, start_time = None):
    ave_cpu_util = 0.0
    for j in range(len(task["nodes"])):
        #print("Hello")
        ave_cpu_util_per_node = 0.0
        count = 0
        for k in range(len(task["nodes"][j]["metrics"]["cpu_util"])):
            if start_time != None and float(task["nodes"][j]["metrics"]["cpu_util"][k][0]) >= float(start_time):
                continue
            ave_cpu_util_per_node += float(task["nodes"][j]["metrics"]["cpu_util"][k][1])
            count += 1
        if count > 0:
            ave_cpu_util_per_node /= count
        ave_cpu_util += ave_cpu_util_per_node
    ave_cpu_util /=len(task["nodes"])
    task["avg_cpu_util"] = ave_cpu_util
    #print(task["task_id"], ave_cpu_util)
    
def calc_avg_gpu_util(task, start_time = None):
    avg_gpu_util = 0.0
    if int(task["gpus"]) > 0:
        for j in range(len(task["nodes"])):
            avg_gpu_util_per_node = 0.0
            count = 0
            for k in range(len(task["nodes"][j]["metrics"]["gpu_util"])):
                if start_time != None and float(task["nodes"][j]["metrics"]["gpu_util"][k][0]) >= float(start_time):
                    continue
                for i in range(1, int(task["gpus"])):
                    avg_gpu_util_per_node += float(task["nodes"][j]["metrics"]["gpu_util"][k][i])
                count += 1
            if count > 0:
                avg_gpu_util_per_node /=(int(task["gpus"])* count)
            avg_gpu_util += avg_gpu_util_per_node
    task["avg_gpu_util"] = avg_gpu_util

def calc_avg_memory_util(task, start_time = None):
    ave_memory_util = 0.0
    for j in range(len(task["nodes"])):
        ave_memory_util_per_node = 0.0
        count = 0
        for k in range(len(task["nodes"][j]["metrics"]["memory_util"])):
            if start_time != None and float(task["nodes"][j]["metrics"]["memory_util"][k][0]) >= float(start_time):
                continue
            ave_memory_util_per_node += float(task["nodes"][j]["metrics"]["memory_util"][k][1])
            count += 1
        if count > 0:
            ave_memory_util_per_node /= count
        ave_memory_util += ave_memory_util_per_node
    ave_memory_util /=len(task["nodes"])
    task["avg_memory_util"] = ave_memory_util

