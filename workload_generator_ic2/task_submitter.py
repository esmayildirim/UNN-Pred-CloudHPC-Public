import task_pooler as tp
import config as cf
import os
import random
import math
import time
import json
import radical.pilot as rp 
from pprint import pprint 
import metrics_withgpu
from scipy.stats import poisson

WORK_DIR = os.path.dirname(os.path.realpath(__file__))
RESOURCE_JSON = "resource_bnl.json"


os.environ['RADICAL_PROFILE'] = 'TRUE'
# prepare resource configuration
os.system(f'mkdir -p {WORK_DIR}/.radical/pilot/configs && '
          f'cd {WORK_DIR} && cp {RESOURCE_JSON} .radical/pilot/configs/')
os.environ['RADICAL_CONFIG_USER_DIR'] = WORK_DIR
# for debug purposes
os.environ['RADICAL_LOG_LVL'] = 'DEBUG'
os.environ['RADICAL_REPORT'] = 'TRUE'

def init(num_nodes, runtime_mins, service_path, venv_path):
    session = rp.Session()
    pmgr    = rp.PilotManager(session=session)
    tmgr    = rp.TaskManager(session=session)
    PILOT_DESCRIPTION = {
    'resource': 'bnl.ic2_csi',  # OR 'bnl.ic2_debug'
    'project': 'csihpc',
    'nodes': num_nodes,
    'runtime': runtime_mins, # Is this total time it takes to execute all tasks submitted?
    'sandbox' : WORK_DIR
    }
    PILOT_DESCRIPTION['services'] = []

    for idx in range(PILOT_DESCRIPTION['nodes']):
        PILOT_DESCRIPTION['services'].append(
            rp.TaskDescription({
                'executable': 'python3',
                'arguments': [service_path+'monitor_threaded.py', runtime_mins],
                'stdout': f'{WORK_DIR}/watcher_{session.uid}_{idx}.txt',
                'ranks': 1,
                'cores_per_rank': 1,
                'tags': {'colocate': str(idx),
                         'exclusive': True},
                'pre_exec': ['hostname', 'module load cuda'
                             #f'source {venv_path}/bin/activate'
                             ]
            })
        )
    pilot = pmgr.submit_pilots(rp.PilotDescription(PILOT_DESCRIPTION))

    tmgr.add_pilots(pilot)
    print("hello")
    pilot.wait(rp.PMGR_ACTIVE)
    print("pilot active")    
    return session, pilot, tmgr


def submit_tasks(session, tmgr, task_desc_list, num_tasks, inter_arrival_times, seed_num): # multiple sessions must be possible
    task_list = []    
    random.seed(seed_num)
    for i in range(num_tasks):
        random_id = random.randrange(len(task_desc_list))
        print("random id:", random_id)
        task = tmgr.submit_tasks(task_desc_list[random_id]['desc'])
        print("Submitting task:", task.description["executable"], task.description['arguments'],
              task.description['ranks'], task.description['cores_per_rank'], flush=True)
        
        #time.sleep(inter_arrival_times[i])
    
        dict_task = {}
        dict_task = metrics_withgpu.create_task(dict_task,
                                                int(str(task.uid).split(".")[1]),
                                                session.uid,
                                                task_desc_list[random_id]['app'],
                                                time.time(),
                                                task.description['ranks']*task.description['cores_per_rank'],
                                                task.description['gpus_per_rank'])  
        task_list.append(dict_task)

    tmgr.wait_tasks()
    session.close(download=True)
    return task_list        

def write_workload_data(task_list, workload_name, platform, pilot_id, num_nodes, output_file): 
    workloads = []
    workload ={}
    workload['workload-name'] = workload_name
    workload['platform'] = platform
    task_list_updated = []
    for task_dict in task_list:
        task_dict = metrics_withgpu.add_start_finish(task_dict, pilot_id)        
        
        #print(task_dict)
        task_dict = metrics_withgpu.add_metrics(task_dict,
                                                metrics_withgpu.task_nodelist(task_dict, 
                                                                    "./"+task_dict['session_id']+"/tmgr.0000.json"),
                                                num_nodes,
                                                pilot_id
                                                )        
        print(task_dict)
        task_list_updated.append(task_dict)
    workload['tasklist'] = task_list_updated
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
             workloads= json.load(f)
    workloads.append(workload)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(workloads, f, ensure_ascii=False, indent=4)

def write_system_load_data(session_id, pilot_id, num_nodes, workload_name, platform, output_file):
    workloads = []
    workload ={}
    workload['workload-name'] = workload_name
    workload['platform'] = platform
    node_list = []
    for file_id in range(num_nodes): 
        node_util = metrics_withgpu.system_load(file_id, session_id, pilot_id)
        node_list.append(node_util)
    workload['node_list'] = node_list
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
             workloads= json.load(f)
    workloads.append(workload)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(workloads, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    #seed_list = [10, 35, 62, 89, 22]
    seed_list = [62, 89, 22]
    #seed = 10
    for seed in seed_list:
        allocated_num_nodes = 2
        allocated_time_mins = 720
        num_tasks_to_submit = 50
        mean_inter_arrival_time_secs = 0
        inter_arrival_times = poisson.rvs(mu=mean_inter_arrival_time_secs, size = num_tasks_to_submit)

        tp.pooler(allocated_time_mins)
        session, pilot, tmgr = init(allocated_num_nodes,
                                allocated_time_mins,
                                "/hpcgpfs01/scratch/eyildirim/workload_generator/",
                                "")
        task_json_list = submit_tasks(session,
                                  tmgr,
                                  tp.task_pool,
                                  num_tasks_to_submit,
                                  inter_arrival_times,
                                  seed)

        with open("task_list_"+session.uid+".json", 'w', encoding='utf-8') as f:
            json.dump(task_json_list, f, ensure_ascii=False, indent=4)

#    with open("task_list_"+"rp.session.ic2submit01.sdcc.bnl.gov.eyildirim.019916.0000"+".json", 'r') as f:
#             task_json_list= json.load(f)
        write_workload_data(task_json_list, 
                        "mixed-tasks"+str(seed),
                        "hpc_ic2",
                        pilot.uid,
                        allocated_num_nodes,
                        "workloads_mixed.json")
    
        write_system_load_data(session.uid,
                           pilot.uid,
                           allocated_num_nodes,
                           "mixed-tasks"+str(seed),
                           "hpc_ic2",
                           "system_load_mixed.json")
    
        print("Finished", seed)
