import radical.pilot as rp

KERAS_APP_PATH = "/home/eyildirim/workload_generator/"

KERAS_DATA_PATH = "/home/eyildirim/"
keras_app_name = "image_classification_wGPU_monitor.py"

def keras_app_task_desc(app_path, args, cores_per_rank = 1, gpus_per_rank = 0):
    arguments = []
    arguments.append(app_path)
    arguments += args
    print(args)
    return rp.TaskDescription({
                        'executable': 'python3',
                        'arguments': arguments,
                        'ranks': 1,
                        'cores_per_rank': cores_per_rank,
                        'gpus_per_rank': gpus_per_rank,
                        'pre_exec': [
                                     #'module use /soft/modulefiles',
                                     #'module load conda',
                                     #'eval "$(conda shell.posix hook)"',
                                     #'conda activate ve.rp3',
                                     #'module load PrgEnv-nvhpc',
                                     #'module load craype-accel-nvidia80',
                                     #'module load cuda',
                                     'export MPICH_GPU_SUPPORT_ENABLED=0',
                            #'export NVIDIA_DIR=$(dirname $(dirname $(python3 -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)")))',
                            #"""export LD_LIBRARY_PATH=$(echo ${NVIDIA_DIR}/*/lib/ | sed -r 's/\s+/:/g')${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"""],
                                     ],
                                     #'pre_launch':['export OMPI_MCA_hwloc_base_binding_policy=none'],
                        'named_env': 'rp'  # reuse virtual env used to launch RP-Agent
                
                        })

