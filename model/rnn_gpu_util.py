import json
import calculate_system_utils as csu
import calculate_task_utils as ctu
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from math import sqrt
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import AveragePooling1D
from keras.layers import MaxPooling1D
from keras.layers import Conv1D
from keras.layers import Flatten
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
def correlation(X, y):
    print("PR\tSR\tKT")
    for i in range(X.shape[1]):
        #print("X col "+str(i)+" max - min ", np.max(X[:,i]),np.min(X[:,i]))
        pr = st.pearsonr(X[:,i], y[:,0])[0] #correlation
        sr = st.spearmanr(X[:,i], y[:,0])[0] # spearman r
        kt = st.kendalltau(X[:,i], y[:,0])[0]
        print("%.2f\t%.2f\t%.2f" % (pr, sr, kt))
        
        #plt.hist(X[:,i])
        #plt.scatter(X[:,i], y[:,0])
        #plt.show()
    print("Y max - min ", np.max(y[:,0]),np.min(y[:,0]))
# prepare dataset with input and output scalers, can be none

def scale_fit(y):
    min = np.min(y[:,0])
    max = np.max(y[:,0])
    return min, max
        
def scale_transform(y, min, max):
    for i in range(y.shape[0]):
        y[i][0] = (y[i][0]-min)/(max-min)
    return y
        
def scale_inv_transform(y, min,max):
    for i in range(y.shape[0]):
        y[i][0] = (y[i][0] * (max-min)) + min
    return y
        
# split dataset into training and testing and normalize
def get_dataset(input_scaler, scale_output, X, y):
    
    correlation(X, y)
    
    #trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.25, random_state=42)
    trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.25)
    
    if input_scaler is not None:
        # fit scaler
        inverse_scaler_X = input_scaler.fit(trainX)
        # transform training dataset
        trainX = input_scaler.transform(trainX)
        # transform test dataset
        testX = input_scaler.transform(testX)
        
    if scale_output == True:
        train_min, train_max = scale_fit(trainy)
        testy = scale_transform(testy, train_min, train_max)
        trainy = scale_transform(trainy, train_min, train_max)
    
    return trainX, trainy, testX, testy, train_min, train_max

# fit and evaluate model on test set
def evaluate_model(trainX, trainy, testX, testy, train_min, train_max):
    # define model
    trainX = trainX.reshape(trainX.shape[0], trainX.shape[1], 1)
    model = Sequential()
    model.add(SimpleRNN(50, input_shape=(trainX.shape[1],trainX.shape[2]), activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(25, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='linear'))
    # compile model
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.01))
    history = model.fit(trainX, trainy, epochs=100, verbose=0, validation_split=0.2)
    # evaluate the model
    
    percentage = 0.0
    mae = 0.0
    mse = 0.0
    count = 0
    
    testX = testX.reshape(testX.shape[0], testX.shape[1], 1)
    y_hat = model.predict(testX)
    testy = scale_inv_transform(testy, train_min, train_max)
    y_hat = scale_inv_transform(y_hat, train_min, train_max)
    
    for i in range(y_hat.shape[0]):
        mae += abs(y_hat[i][0] - testy[i][0])
        mse += abs(y_hat[i][0] - testy[i][0])**2
        #if testy[i][0] == 0:
        #    percentage += abs(y_hat[i][0] - testy[i][0])
        #else: percentage += abs(y_hat[i][0] - testy[i][0])/ testy[i][0]
        count += 1
    mae /= count
    mse /= count
    #percentage /= count
    #percentage *= 100
    r2 = r2_score(testy[:,0],y_hat[:,0])
    #print("MAPE:", percentage)
    print("MSE:", mse)
    print("MAE:", mae)
    print("R2:", r2)
    print("RMSE:", sqrt(mse))
    
    #plt.scatter(testy[:,0],y_hat[:,0], facecolors='none', edgecolors='b')
    #plt.plot([0, np.max(testy[:,0])], [0, np.max(testy[:,0])], color = 'red', linewidth = 1)
    #plt.show()
    
    return model, mae, mse, r2, percentage
# read dataset files and convert into proper format
def read_workloads_data(file_path, file_path2, file_path3, file_path4,total_cpus1, total_gpus1, total_cpus2, total_gpus2, util_type = "cpu"):
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
            
            input_item.append(float(task["avg_cpu_util"])) #4
            input_item.append(float(task["avg_gpu_util"])) #5
            
            input_item.append(float(task["avg_memory_util"])) #6
            
            input_item.append(float(task["avg_network_in_Mbps"])) #7
            #input_item.append(float(task["avg_network_out_Mbps"])) #8
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
            
            
            # second system tasks
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
                        #other_run_time = other_finish_time - other_start_time
                        
                        #cpu_utilization
                        if util_type == "cpu":
                            #other_output_util = float(csu.calc_node_cpu_util(other_node_list, other_start_time, other_finish_time))
                            ctu.calc_avg_cpu_util(other_task)
                            other_output_util = float(other_task["avg_cpu_util"])
                        else:
                            #other_output_util = float(csu.calc_node_gpu_util(other_node_list, other_start_time, other_finish_time))
                            ctu.calc_avg_gpu_util(other_task)
                            other_output_util = float(other_task["avg_gpu_util"])
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
                        output_list.append([other_output_util])
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
    RUNS = 10
    '''
    X, y = read_workloads_data("../data/all_workloads_aws.json",
                        "../data/all_system_loads_aws.json",
                        "../data/all_workloads_ic2.json",
                        "../data/all_system_loads_ic2.json",
                        128, 0, 96, 8,  "gpu")
    
    X, y = read_workloads_data("../data/all_workloads_aws.json",
                        "../data/all_system_loads_aws.json",
                        "../data/all_workloads_polaris.json",
                        "../data/all_system_loads_polaris.json",
                        128, 0, 320, 40,  "gpu")
    '''
    X, y = read_workloads_data("../data/all_workloads_polaris.json",
                        "../data/all_system_loads_polaris.json",
                        "../data/all_workloads_ic2.json",
                        "../data/all_system_loads_ic2.json",
                        320, 40, 96, 8, "gpu")
    
    #calculate average of 10 runs
    
    averages = [0.0, 0.0, 0.0, 0.0]
    for i in range(RUNS):
        trainX, trainy, testX, testy, trainy_min, trainy_max = get_dataset(MinMaxScaler(), True, X, y)
        model, mae, mse, r2, mape = evaluate_model(trainX, trainy, testX, testy, trainy_min, trainy_max)
        averages[0] += mae
        averages[1] += mse
        #averages[2] += mape
        averages[3] += r2
    
    for i in range(len(averages)):
        averages[i] /= RUNS
    
    print("AVG_MAE:", averages[0])
    print("AVG_MSE", averages[1])
    #print("AVG_MAPE", averages[2])
    print("AVG_R2", averages[3])
    
