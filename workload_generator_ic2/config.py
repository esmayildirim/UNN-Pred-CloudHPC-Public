
config = { "nas":
            {   "paradigm"  : ["openmp","mpi","hybrid"], #["mpi", "openmp", "hybrid"],
                "class"     : ["C","D"], # ["C", "D"],
                "openmp"    : {
                    "max_threads" : 48
                    },
                "mpi"       : {
                    "nodes" : 2
                    },
                "hybrid"    : {
                    "nodes" : 2
                    },
                "iterations": 1
             }
            ,
            
            "image_classification":
            {   
                 "epochs"    : [1, 5],
                 "data_size" : ["small", "medium", "large"], # medium and large are also possibilites
                 "batch_size": [32, 64],
                 "iterations": 2,
                 "gpus_per_rank": [1,2,4]
            }
          }

