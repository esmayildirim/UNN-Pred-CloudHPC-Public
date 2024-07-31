#This module will include functions to parse metrics and create a json format for the jobs
import pprint
import json
#This function to be called rp program during submission
def create_task(task_dict, taskid,session_id, app_name, submit_time, cpus, gpus = 0):
    task_dict["task_id"] = taskid
    task_dict["session_id"] = session_id
    task_dict["app_name"] = app_name
    task_dict["submit_time"] = submit_time
    task_dict["cpus"] = cpus
    task_dict["gpus"] = gpus 
    pprint.pp(task_dict, indent=4)
    return task_dict 

#This function to be called after the workload is executed
def add_start_finish(task, pilot_id):
    #session = 'rp.session.ic2submit01.sdcc.bnl.gov.eyildirim.019879.0002'
    #pilot = 'pilot.0000'
    taskid = "task."+ "0" * (6-len(str(task['task_id']))) + str(task['task_id'])
    task_path = './radical.pilot.sandbox/'+ str(task['session_id']) + '/' + pilot_id + '/' + taskid + '/'+ taskid + '.prof'
    f2 = open(task_path, 'r')
    lines = f2.readlines()
    for line in lines:
        if 'rank_start' in line:
            stimestamp = line.split(',')[0]
            break
    for line in lines:
        if 'rank_stop' in line:
            etimestamp = line.split(',')[0]
        #print(stimestamp + '\t'+etimestamp)
    task['start_time'] = stimestamp
    task['finish_time'] = etimestamp
    return task

#this function is to be called after the workload finishes, parses tmgr json file to gather task nodelist
def task_nodelist(task, tmgr_json_path):
    f = open(tmgr_json_path)
    data = json.load(f)
    task_list = list(data['tasks'].keys())
    node_list = set()
    #print("TASKID\t#CPUS\tNODELIST\tSTART TIME\tEND TIME")
    for taskid in task_list:
        if taskid.endswith(str(task['task_id'])):
            taskidint = int(taskid.split('.')[1])
            #print(taskidint, end='\t')
            num_cpus = data['tasks'][taskid]['resources']['cpu']
            node_list = set([rank['node_name'] for rank in data['tasks'][taskid]['slots']['ranks']])
            #print(num_cpus, end = '\t')
            print("NODELIST",node_list, end = '\t')
            return node_list

#this function is to be called after the workload finishes, parses watcher files to gather resource metrics 
def add_metrics(task, node_list, num_all_nodes, pilot_id):
    dict_list = []
    watcher_ids = []
    for i in range(num_all_nodes):
       f = open("watcher_"+ task['session_id'] + "_" + str(i)+ ".txt", "r")
       node_name = f.readline()
       if node_name.strip() in node_list:
           watcher_ids.append(i)
       f.close()

    for watcher_id in watcher_ids: #NOT THE WATCHER ID - REWRITE 

        f = open("watcher_"+ task['session_id'] + "_" + str(watcher_id)+ ".txt","r")
        lines = f.readlines()
        #print(node_list, flush=True)        
        if len(lines)>0 and lines[0].strip() in node_list:
            cpu_list=[]
            disk_list=[]
            network_list=[]
            memory_list=[]
            for i in range(1, len(lines)):
                if lines[i].startswith("CPU:"):
                    cpu_metrics = lines[i].split()
                    if float(cpu_metrics[1]) >= float(task["start_time"]) and float(cpu_metrics[1]) <= float(task["finish_time"]):
                        cpu_list.append((cpu_metrics[1], cpu_metrics[2][:-1])) #timestamp, cpu utilization
                if lines[i].startswith("MEMORY:"):
                    memory_metrics = lines[i].split()
                    if float(memory_metrics[1]) >= float(task["start_time"]) and float(memory_metrics[1]) <= float(task["finish_time"]):
                        memory_list.append((memory_metrics[1], memory_metrics[2][:-1])) #timestamp, memory utilization
                if lines[i].startswith("DISK:"):
                    disk_metrics = lines[i].split()
                    if float(disk_metrics[1]) >= float(task["start_time"]) and float(disk_metrics[1]) <= float(task["finish_time"]):
                        disk_list.append((disk_metrics[1], disk_metrics[2],disk_metrics[3]))
                if lines[i].startswith("NETWORK:"):
                    network_metrics = lines[i].split()
                    if float(network_metrics[1]) >= float(task["start_time"]) and float(network_metrics[1]) <= float(task["finish_time"]):
                        network_list.append((network_metrics[1], network_metrics[2], network_metrics[3]))
            node_name = lines[0].strip()
            dict= {}
            dict["node_name"] = node_name
            dict["metrics"]= {}
            dict["metrics"]["cpu_util"]= cpu_list
            dict["metrics"]["network_util"]= network_list
            dict["metrics"]["disk_util"]= disk_list
            dict["metrics"]["memory_util"] = memory_list
            #GPU UTIL
            #pilot = 'pilot.0000'
            taskid = "task."+ "0" * (6-len(str(task['task_id']))) + str(task['task_id'])
            task_path = './radical.pilot.sandbox/'+ str(task['session_id']) + '/' + pilot_id + '/' + taskid + '/'+ taskid + '.out'
            f2 = open(task_path, 'r')
            lines = f2.readlines()
            gpu_list = []
            for line in lines:
                if line.startswith('GPU'):
                    wordlist =line.split()
                    del wordlist[0]
                    zero_flag = True
                    for i in range(len(wordlist)):
                        if wordlist[i].endswith('%'):
                            wordlist[i] = wordlist[i][:-1]
                            if float(wordlist[i]) > 0.0:
                                zero_flag = False
                    if not zero_flag:
                        gpu_list.append(wordlist)        
            dict["metrics"]["gpu_util"] = gpu_list
            f2.close()
            dict_list.append(dict)
            f.close()
    task["nodes"] = dict_list
        
        #pprint.pp(task, indent=4)
    return task 

def system_load(watcher_id, session_id, pilot_id):
    f = open("watcher_"+ session_id + "_" + str(watcher_id)+ ".txt","r")
    lines = f.readlines()
    dict = {}
    if len(lines)>0:
        cpu_list=[]
        disk_list=[]
        network_list=[]
        memory_list=[]
        for i in range(1, len(lines)):
            if lines[i].startswith("CPU:"):
                cpu_metrics = lines[i].split()
                cpu_list.append((cpu_metrics[1], cpu_metrics[2][:-1])) #timestamp, cpu utilization
            if lines[i].startswith("MEMORY:"):
                memory_metrics = lines[i].split()
                memory_list.append((memory_metrics[1], memory_metrics[2][:-1])) #timestamp, memory utilization
            if lines[i].startswith("DISK:"):
                disk_metrics = lines[i].split()
                disk_list.append((disk_metrics[1], disk_metrics[2],disk_metrics[3]))
            if lines[i].startswith("NETWORK:"):
                network_metrics = lines[i].split()
                network_list.append((network_metrics[1], network_metrics[2], network_metrics[3]))
        node_name = lines[0].strip()
        dict["node_name"] = node_name
        dict["metrics"]= {}
        dict["metrics"]["cpu_util"]= cpu_list
        dict["metrics"]["network_util"]= network_list
        dict["metrics"]["disk_util"]= disk_list
        dict["metrics"]["memory_util"]= memory_list
    #GPU UTIL
    #find the tasks that were executed on this node from tmgr.0000.json
    task_ids_on_node = []
    tmgr_json_path = session_id + "/tmgr.0000.json";
    f = open(tmgr_json_path)
    data = json.load(f)
    task_list = list(data['tasks'].keys())
    node_list = set()
    for taskid in task_list:
        node_list = set([rank['node_name'] for rank in data['tasks'][taskid]['slots']['ranks']])
        for node_n in node_list:
            if node_n == dict["node_name"]:
                task_ids_on_node.append(taskid)
                break
    f.close()
    #print(task_ids_on_node)
    #access the pilot folder and tasks output files to gather gpu util info
    gpu_list=[]
    for idx in task_ids_on_node:
        path = "radical.pilot.sandbox/" + session_id + "/"+ pilot_id+ "/"+ idx+ "/"+idx+".out"
        f2 = open(path, 'r')
        lines = f2.readlines()
        
        for line in lines:
            if line.startswith('GPU'):
              #  print(line)
                wordlist =line.split()
                del wordlist[0]
                zero_flag = True
                for i in range(len(wordlist)):
                    if wordlist[i].endswith('%'):
             #           print(wordlist[i])
                        wordlist[i] = wordlist[i][:-1]
                        
                        if float(wordlist[i]) > 0.0:
              #              print("not zero")
                            zero_flag = False
              #  print([metric[0] for metric in gpu_list])
                if wordlist[0] in [metric[0] for metric in gpu_list]:
                    continue
                if not zero_flag:
            #        print("adding", wordlist)
                    gpu_list.append(wordlist)
        f2.close()
    dict["metrics"]["gpu_util"] = gpu_list
    return dict

if __name__ == "__main__":
    for i in range(2):
        print(system_load(i, "rp.session.polaris-login-04.eyildirim.019906.0006", "pilot.0000")["metrics"]["gpu_util"])

    '''
    f = open("rp.session.ip-10-0-0-233.ec2-user.019901.0006/tmgr.0000.json")
    data = json.load(f)
    task_list = list(data['tasks'].keys())
    node_list = set()
    #print("TASKID\t#CPUS\tNODELIST\tSTART TIME\tEND TIME")
    for taskid in task_list:
        for i in range(5):
            if taskid.endswith(str(i)):
                taskidint = int(taskid.split('.')[1])
                print(taskidint, end='\t')
                num_cpus = data['tasks'][taskid]['resources']['cpu']
                node_list = set([rank['node_name'] for rank in data['tasks'][taskid]['slots']['ranks']])
                print(num_cpus, end = '\t')
                print(node_list, end = '\n')
        #return node_list
    ''' 


