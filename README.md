# UNN-Pred-CloudHPC-Public
This project aims to develop neural network models for predicting job runtime and resource utilization on integrated cloud and  hpc systems. Given the run time of the job on one system and the load of both systems, the model predicts the runtime of the job on the other system. 

## Workload Generator

The main program is task_submitter.py and it is used to run workloads using Radical Pilot software and collect data about their resource utilization.


## Data

workloads and system_load files contain time series data about the tasks/jobs that belong to a workload and the system utilization of the allocated nodes the workload applications had been running. Due to the large size of the dataset, it is available in Zenodo platform:

https://doi.org/10.5281/zenodo.15545095


## Model
Using the data in the data folder, we have three MLP models to predict runtime, cpu utilization and gpu utilization. This folder also contains comparison models (CNN, Linear Regression, RNN and LSTM).

## Visualization

Contains programs to visualize the data collected in the Data folder

## Project Team
Esma Yildirim, eyildirim@qcc.cuny.edu

Mohab Hussein, mohab.hussein95@qmail.cuny.edu

Mikhail Titov, mtitov@bnl.gov

Ozgur Kilic, okilic@bnl.gov
