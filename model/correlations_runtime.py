import json
import calculate_system_utils as csu
import calculate_task_utils as ctu
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from math import sqrt
# compare scaling methods for mlp inputs on regression problem
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from numpy import mean
from numpy import std
from sklearn.metrics import r2_score
from scipy.stats import chi2_contingency
import scipy.stats as st
import pandas as pd
from keras.models import model_from_json
import threading
FIELDS = [ "Runtime on Sys1",
           "# of CPUs on Sys1",
           "# of GPUs on Sys1",
           "CPU Util of Task on Sys1",
           "GPU Util of Task on Sys1",
           "Memory Util of Task on Sys1",
           "Network In (Mbps) on Sys1",
           "Network Out (Mbps) on Sys1",
           "Disk Read (Mbps) on Sys1",
           "Disk Write (Mbps) on Sys1",
           "CPU Util of Sys1 5mins before",
           "GPU Util of Sys1 5mins before",
           "Memory Util of Sys1 5mins before",
           "Network In (Mbps) of Sys1 5mins before",
           "Network Out (Mbps) of Sys1 5mins before",
           "Disk Read (Mbps) of Sys1 5mins before",
           "Disk Write (Mbps) of Sys1 5mins before",
           "CPU Util of Sys1 during task run",
           "GPU Util of Sys1 during task run",
           "Memory Util of Sys1 during task run",
           "Network In (Mbps) of Sys1 during task run",
           "Network Out (Mbps) of Sys1 during task run",
           "Disk Read (Mbps) of Sys1 during task run",
           "Disk Write (Mbps) of Sys1 during task run",
           "Total #CPUs on Sys1",
           "Total #GPUS on Sys1",
           "# of CPUs on Sys2",
           "# of GPUs on Sys2",
           "Total #CPUs on Sys2",
           "Total #GPUS on Sys2",
           "CPU Util of Sys2 5mins before",
           "GPU Util of Sys2 5mins before",
           "Memory Util of Sys2 5mins before",
           "Network In (Mbps) of Sys2 5mins before",
           "Network Out (Mbps) of Sys2 5mins before",
           "Disk Read (Mbps) of Sys2 5mins before",
           "Disk Write (Mbps) of Sys2 5mins before"
         ]

def correlation(X, y, index):
    #print("FIELD\tPearsonR\tSpearmanR\tKendallTau\n")
    #for i in range(X.shape[1]):
        #print("X col "+str(i)+" max - min ", np.max(X[:,i]),np.min(X[:,i]))
        pr = st.pearsonr(X[:,index], y[:,0])[0] #correlation
        sr = st.spearmanr(X[:,index], y[:,0])[0] # spearman r
        kt = st.kendalltau(X[:,index], y[:,0])[0]
        print(FIELDS[index]+":", end = "", flush=True)
        print("%.2f\t:%.2f\t:%.2f" % (pr, sr, kt), flush=True)
        
        #plt.hist(X[:,i])
        #plt.scatter(X[:,i], y[:,0])
        #plt.show()
    #print("Y max - min ", np.max(y[:,0]),np.min(y[:,0]))
# prepare dataset with input and output scalers, can be none


        


def read_workloads_data(file_path, file_path2, file_path3, file_path4,total_cpus1, total_gpus1, total_cpus2, total_gpus2):
    input_list = []
    output_list = []
    with open(file_path, 'r') as f1:
        workloads= json.load(f1)
    #app_names = set()
    with open(file_path2, 'r') as f2:
        system_loads= json.load(f2)
    #task_dict = {}
    
    with open(file_path3, 'r') as f3:
        second_workloads = json.load(f3)
    with open(file_path4, 'r') as f4:
        second_system_loads = json.load(f4)
    
    for i in range(len(workloads)):
        #print(len(workloads), " workload, ", len(workloads[i]["tasklist"]), " tasks.")
        tasks = workloads[i]["tasklist"]
        #app_names |= set([task["app_name"] for task in tasks])
        node_list = system_loads[i]["node_list"]
        for task in tasks:
            #print("Prep")
            input_item = [] # input layer
            start_time = float(task["start_time"])
            finish_time = float(task["finish_time"])
            run_time = finish_time - start_time
            #calculate average cpu util for
            task["run_time"] = run_time
            input_item.append(run_time) #1 runtime
            input_item.append(float(task["cpus"])) #2 cpus
            input_item.append(float(task["gpus"])) #3 gpus
            ctu.calc_avg_cpu_util(task)
            ctu.calc_avg_gpu_util(task)
            ctu.calc_avg_memory_util(task)
            ctu.calc_avg_network_util(task)
            ctu.calc_avg_disk_util(task)
            #csu.calc_node_gpu_util(node_list, start_time, finish_time)
            
            #print("TASK", task["task_id"])
            #print("Tasks utilizations:", end = " ")
            #print("%.2f %.2f %.2f %.2f %.2f %.2f %.2f" % (task["avg_cpu_util"], task["avg_gpu_util"],
            #      task["avg_memory_util"],task["avg_network_in_kBps"], task["avg_network_out_kBps"],
            #        task["avg_disk_read_kBps"], task["avg_disk_write_kBps"]))
            #add the task utilizations
            
            input_item.append(float(task["avg_cpu_util"])) #4
            input_item.append(float(task["avg_gpu_util"])) #5
            
            input_item.append(float(task["avg_memory_util"])) #6
            input_item.append(float(task["avg_network_in_Mbps"])) #7
            input_item.append(float(task["avg_network_out_Mbps"])) #8
            input_item.append(float(task["avg_disk_read_Mbps"])) #9
            input_item.append(float(task["avg_disk_write_Mbps"])) #10
            
            #add the system load 5 minutes before the app started.
            system_utils = []
            system_utils.append(float(csu.calc_node_cpu_util(node_list, start_time - 300, start_time))) #11
            system_utils.append(float(csu.calc_node_gpu_util(node_list, start_time - 300, start_time))) # 12
            system_utils.append(float(csu.calc_node_memory_util(node_list, start_time - 300, start_time))) #13
            system_utils.append(float(csu.calc_node_network_util(node_list, start_time - 300, start_time, 1))) #14
            system_utils.append(float(csu.calc_node_network_util(node_list, start_time - 300, start_time, 2))) #15
            system_utils.append(float(csu.calc_node_disk_util(node_list, start_time -300 , start_time, 1))) #16
            system_utils.append(float(csu.calc_node_disk_util(node_list, start_time -300, start_time, 2))) #17
            input_item += system_utils
            #add the system load during the apps running time
            system_utils = []
            system_utils.append(float(csu.calc_node_cpu_util(node_list, start_time, finish_time))) #18
            system_utils.append(float(csu.calc_node_gpu_util(node_list, start_time, finish_time))) #19
            system_utils.append(float(csu.calc_node_memory_util(node_list, start_time, finish_time))) #20
            system_utils.append(float(csu.calc_node_network_util(node_list, start_time, finish_time, 1))) #21
            system_utils.append(float(csu.calc_node_network_util(node_list, start_time, finish_time, 2))) #22
            system_utils.append(float(csu.calc_node_disk_util(node_list, start_time, finish_time, 1))) #23
            system_utils.append(float(csu.calc_node_disk_util(node_list, start_time, finish_time, 2))) #24
            #add the system load during the app's run
            input_item += system_utils
            
            input_item.append(float(total_cpus1)) # total number of cpus on system 1
            input_item.append(float(total_gpus1))  # total number of gpus on system 1
            
            #print("Average system load during task run:", end=" ")
            ##print (["%0.2f" % i for i in system_utils])
            
            #TODO second system tasks
            # for every same task on the second system calculate 5 minute before system utils and return them
            for j in range(len(second_workloads)):
                #print(len(second_workloads), " workload, ", len(second_workloads[i]["tasklist"]), " tasks.")
                other_tasks = second_workloads[j]["tasklist"]
                #app_names |= set([task["app_name"] for task in tasks])
                other_node_list = second_system_loads[j]["node_list"]
                
                for other_task in other_tasks:
                    temp_item = []
                    if other_task["app_name"] == task ["app_name"]:
                        temp_item.append(float(other_task["cpus"]))# add requested cpu #25
                        temp_item.append(float(other_task["gpus"]))# add requested gpu #26
                        temp_item.append(float(total_cpus2))
                        temp_item.append(float(total_gpus2))
                        
                        #print("Found a similar task:")
                        other_start_time = float(other_task["start_time"])
                        other_finish_time = float(other_task["finish_time"])
                        other_run_time = other_finish_time - other_start_time
                        #add the system utils for 5 minutes before the start time of the task on the second system
                        
                        system_utils_5mins_before = []
                        
                        system_utils_5mins_before.append(float(csu.calc_node_cpu_util(other_node_list, other_start_time - 300, other_start_time))) #27
                        system_utils_5mins_before.append(float(csu.calc_node_gpu_util(other_node_list, other_start_time - 300, other_start_time))) #28
                        system_utils_5mins_before.append(float(csu.calc_node_memory_util(other_node_list, other_start_time- 300, other_start_time))) #29
                        system_utils_5mins_before.append(float(csu.calc_node_network_util(other_node_list, other_start_time- 300, other_start_time, 1))) #30
                        system_utils_5mins_before.append(float(csu.calc_node_network_util(other_node_list, other_start_time- 300, other_start_time, 2))) #31
                        system_utils_5mins_before.append(float(csu.calc_node_disk_util(other_node_list, other_start_time- 300, other_start_time, 1))) #32
                        system_utils_5mins_before.append(float(csu.calc_node_disk_util(other_node_list, other_start_time- 300, other_start_time, 2))) #33
                        
                        temp_item = input_item + temp_item + system_utils_5mins_before
                        
                        input_list.append(temp_item)
                        output_list.append([other_run_time])
                        #print("Average system load in 5 minutes before the task starts ", end = " ")
                        #print (["%0.2f" % i for i in system_utils_5mins_before])
    np.set_printoptions(suppress=True,precision=5)

    input_list = np.array(input_list)
    output_list = np.array(output_list)
    print(input_list.shape)
    print("INPUT LIST")
    print(input_list)
    print(output_list.shape)
    print("OUTPUT LIST")
    print(output_list)
    return input_list, output_list


if __name__ == "__main__":
    #BEST RESULTS
    '''
    X, y = read_workloads_data("../data/ic2/workloads_threaded_new.json",
                        "../data/ic2/system_load_threaded_new.json",
                        "../data/aws/workloads_threaded.json",
                        "../data/aws/system_load_threaded.json",
                        96, 8, 128, 0)
    
    X, y = read_workloads_data("../data/ic2/workloads_small.json",
                        "../data/ic2/system_load_small.json",
                        "../data/aws/workloads_small_5seeds_clean.json",
                        "../data/aws/system_load_small_5seeds.json",
                        96, 8, 128, 0)
    '''
    #MIXED RESULTS
    X, y = read_workloads_data("../data/all_workloads_polaris.json",
                        "../data/all_system_loads_polaris.json",
                        "../data/all_workloads_aws.json",
                        "../data/all_system_loads_aws.json",
                        320, 40, 128, 0)
    
    #threads = []
    print("FIELD\tPearsonR\tSpearmanR\tKendallTau\n")
    for i in range(X.shape[1]):
         correlation(X,y,i)
    #    threads.append(threading.Thread(target=correlation, args=(X,y,i)))
    #    threads[i].start()
    #for i in range(X.shape[1]):
    #    threads[i].join()
            
        
            
      
