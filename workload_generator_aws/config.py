
config = { "nas":
            {   "paradigm"  : ["mpi", "openmp", "hybrid"],
                "class"     : ["C", "D"], # ["C", "D"]
                "openmp"    : {
                    "max_threads" : 32
                    },
                "mpi"       : {
                    "nodes" : 4
                    },
                "hybrid"    : {
                    "nodes" : 4
                    },
                "iterations": 3
             },
            
            "image_classification":
            {   
                 "epochs"    : [1, 5],
                 "data_size" : ["small", "medium", "large"], # medium and large are also possibilites
                 "batch_size": [32, 64],
                 "iterations": 25
            }
          }

