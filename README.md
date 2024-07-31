# UNN-Pred-CloudHPC-Public
This project aims to develop neural network models for predicting job runtime and resource utilization on integrated cloud and  hpc systems. Given the run time of the job on one system and the load of both systems, the model predicts the runtime of the job on the other system. 

## Workload Generator

The main program is task_submitter.py and it is used to run workloads using Radical Pilot software and collect data about their resource utilization.


## Data

workloads and system_load files contain time series data about the tasks/jobs that belong to a workload and the system utilization of the allocated nodes the workload applications had been running. 

## Model
Using the data in the data folder, we have three models to predict runtime, cpu utilization and gpu utilization

## Visualization

Contains programs to visualize the data collected in the Data folder

