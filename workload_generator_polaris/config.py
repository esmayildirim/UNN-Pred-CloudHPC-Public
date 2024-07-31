
config = { "nas":
            {   "paradigm"  : ["openmp","mpi", "hybrid"], #["mpi", "openmp", "hybrid"],
                "class"     : ["C", "D"], # ["C", "D"],
                "openmp"    : {
                    "max_threads" : 32
                    },
                "mpi"       : {
                    "nodes" : 100
                    },
                "hybrid"    : {
                    "nodes" : 100
                    },
                "iterations": 2
             },
             
            
            "image_classification":
            {   
                 "epochs"    : [1, 5],
                 "data_size" : ["small", "medium", "large"], # medium and large are also possibilites
                 "batch_size": [32, 64],
                 "iterations": 10,
                 "gpus_per_rank": [4]
            }
          }

