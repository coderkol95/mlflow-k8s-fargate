# MLOps platform running on Kubernetes

## Introduction

This project aims to be a MLOPS template for UI friendly deep learning development using Optuna and Pytorch, and deployed easily on a K8s cluster.
. This project demonstrates a MLOps application running on Kubernetes with these features
* Preprocessed training data is directly taken from AWS S3 path given by user.
* A artificial neural network trains on the data using [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/).
* Automatic hyperparameter tuning is done using [Optuna](https://optuna.org/).
* Lifecycle management of the project is done using [mlFlow](https://mlflow.org/).
* All the operations are managed through a web app built using Flask, HTML and CSS.
* The web app is containerized and runs on Kubernetes

## Advantages of this approach

* Simple UI based controls for model development, experiment analysis and model registration
* Dockerization ensures portability, repeatability and scalability
* Optuna automatically finds the best hyperparameters [Hyperparameter ranges are supplied via the UI]
* Automatic scaling via Kubernetes
* Stateful data fetching and experiment run saving from/to S3


# Application snapshots

# Running this

# Future work
