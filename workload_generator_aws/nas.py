import radical.pilot as rp

MPI_PATH = "/home/ec2-user/NPB3.4.3/NPB3.4-MPI/bin/"
OMP_PATH = "/home/ec2-user/NPB3.4.3/NPB3.4-OMP/bin/"
HYBRID_PATH = "/home/ec2-user/NPB3.4.3-MZ/NPB3.4-MZ-MPI/bin/"

mpi_apps = ["bt", "cg", "ep", "ft", "is", "lu", "mg", "sp"]
omp_apps = ["bt", "cg", "ep", "ft", "lu", "mg", "sp", "ua"]
hybrid_apps = ["bt-mz", "lu-mz", "sp-mz"]

def mpi_task_desc(app_path, ranks):
    return rp.TaskDescription({
                        'executable': app_path,
                        'arguments': [],
                        'ranks': ranks,
                        'cores_per_rank': 1,
                    })

def omp_task_desc(app_path, cores_per_rank):
    return rp.TaskDescription({
                        'executable': app_path,
                        'arguments': [],
                        'ranks': 1,
                        'cores_per_rank': cores_per_rank,
                        'threading_type': rp.OpenMP,
                        'pre_launch':['export OMPI_MCA_hwloc_base_binding_policy=none']
                    })

def hybrid_task_desc(app_path, ranks, cores_per_rank):
    return rp.TaskDescription({
                        'executable': app_path,
                        'arguments': [],
                        'ranks': ranks,
                        'cores_per_rank': cores_per_rank,
                        'threading_type': rp.OpenMP,
                        'pre_launch':['export OMPI_MCA_hwloc_base_binding_policy=none']
                    })

