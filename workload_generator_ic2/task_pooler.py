import config
import nas
import keras_app
import radical.pilot as rp
import math
from pprint import pprint

max_cores = 47 # 1 core per node is used for service tasks 
task_pool = []
def nas_pooler():
  cf = config.config
  if "nas" in cf.keys():
      for pdg in cf["nas"]["paradigm"]:
          if pdg == "mpi":
                for app in nas.mpi_apps:
                  ranks_possible = []
                  rank_max = max_cores * cf["nas"]["mpi"]["nodes"]
                  if app == "sp" or app == "bt": # must be a square 
                      for i in range (1, int(math.sqrt(rank_max))):
                            ranks_possible.append(i*i)
                  elif app == "lu": # n1 * n2 such that n1/2<=n2 <= n1
                      n = rank_max // 3
                      while n > 0:
                        if 2 * n * n < rank_max:
                            ranks_possible.append(2 *n * n)
                        n //=2
                  #elif app == "ep":
                  #      for i in range(1,rank_max+1):
                  #          ranks_possible.append(i)
                  else:
                        j = 1
                        while (j < rank_max):
                            ranks_possible.append(j)
                            j *= 2
                  for cl in cf["nas"]["class"]:
                        app_path = nas.MPI_PATH + app +"."+cl+".x"
                        for rank in ranks_possible:
                            tdict = {"app": "mpi-"+app+"."+cl+".x", "desc":nas.mpi_task_desc(app_path, rank)}
                            if cl == "D" and rank < max_cores: #make sure D class tasks do not share memory with others 
                                continue
                            if cl == "D" and (app == "ft" or app == "is"): #make sure ft.D.x does not run 
                                continue
                            print(app_path, rank)
                            for iter in range(int(cf["nas"]["iterations"])):
                                task_pool.append(tdict)
          elif pdg == "openmp":
              for app in nas.omp_apps:
                  rank_max = min(max_cores, cf["nas"]["openmp"]["max_threads"])
                  i = 1
                  while i <= rank_max:
                      for cl in cf["nas"]["class"]:
                        app_path = nas.OMP_PATH + app + "."+cl+ ".x"
                        tdict = {"app": "omp-"+app+"."+cl+".x", "desc":nas.omp_task_desc(app_path, i)}
                        if cl == "D": #make sure D class tasks do not share memory with others 
                            tdict = {"app": "omp-"+app+"."+cl+".x", "desc":nas.omp_task_desc(app_path, max_cores)}
                        if cl == "D" and (app == "ft" or app == "is"): #make sure ft.D.x does not run 
                            continue
                        if cl == "D":
                            print(app_path, max_cores)
                        else:  print(app_path, i) 
                        for iter in range(int(cf["nas"]["iterations"])):
                            task_pool.append(tdict)
                        #task_pool.append(nas.omp_task_desc(app_path, i))
                      i *= 2
          elif pdg == "hybrid":
              for app in nas.hybrid_apps:
                  rank_max = cf["nas"]["hybrid"]["nodes"]
                  i = 1
                  while i <= rank_max:
                    for cl in cf["nas"]["class"]:
                        app_path = nas.HYBRID_PATH + app + "." + cl + ".x"
                        tdict = {"app": app+"."+cl+".x", "desc":nas.hybrid_task_desc(app_path, i, max_cores)}
                        if cl == "D" and (app == "ft" or app == "is"): #make sure ft.D.x does not run 
                            continue
                        print(app_path, i, "ranks", max_cores, "cores per rank")
                        for iter in range(int(cf["nas"]["iterations"])):
                            task_pool.append(tdict)
                    i *= 2
  print(len(task_pool), "tasks generated")    
def keras_pooler(runtime_mins):
    
    cf = config.config

    if "image_classification" in cf.keys():
        app_path = keras_app.KERAS_APP_PATH + "/" + keras_app.keras_app_name
        for bs in cf["image_classification"]["batch_size"]:
            for ds in cf["image_classification"]['data_size']:
                for ep in cf["image_classification"]['epochs']:
                    for gpr in cf["image_classification"]["gpus_per_rank"]:
                        fdict = {"app": keras_app.keras_app_name+ "_bs"+str(bs) + "_ds"+ds+ "_ep"+str(ep),
                             "desc": keras_app.keras_app_task_desc(app_path, 
                                    [keras_app.KERAS_DATA_PATH,bs, ds,ep,runtime_mins],
                                    1, 
                                    gpr)}

                        print(keras_app.keras_app_name+ "_bs"+str(bs) + "_ds"+ds+ "_ep"+str(ep), 1, gpr)
                        for iter in range(int(cf["image_classification"]["iterations"])):
                            task_pool.append(fdict)
    print(len(task_pool), "tasks_generated")
def pooler(runtime_mins):
    nas_pooler()
    keras_pooler(runtime_mins)

#task_pool.append(nas.hybrid_task_desc(app_path, i, max_cores))
if __name__=="__main__":
    #pprint(config.config, indent=4)
    pooler(20)
    #print(len(task_pool), "tasks generated")

