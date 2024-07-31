import radical.pilot as rp

KERAS_APP_PATH = "/home/ec2-user/workload_generator/"

KERAS_DATA_PATH = "/home/ec2-user/workload_generator/"
keras_app_name = "image_classification_wGPU_monitor.py"


def keras_app_task_desc(app_path, args, cores_per_rank, gpus_per_rank = 0):
    arguments = []
    arguments.append(app_path)
    arguments += args
    print(args)
    return rp.TaskDescription({
                        'executable': 'python3.9',
                        'arguments': arguments,
                        'ranks': 1,
                        'cores_per_rank': cores_per_rank,
                        #'gpus_per_rank': gpus_per_rank,
                        'pre_launch':['export OMPI_MCA_hwloc_base_binding_policy=none'],
                        'named_env': 'rp'  # reuse virtual env used to launch RP-Agent
                
                        })

