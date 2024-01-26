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

## Landing page
<img width="1439" alt="image" src="https://github.com/coderkol95/mlflow-optuna-k8s/assets/15844821/a26e3888-9908-42f4-b7b2-ecf9a8007e1e">

## Model training page
<img width="1438" alt="image" src="https://github.com/coderkol95/mlflow-optuna-k8s/assets/15844821/f0f54221-6685-4c35-b558-3cc64e8e450f">

## Experiment analysis journey
### 1. Select the date
<img width="1438" alt="image" src="https://github.com/coderkol95/mlflow-optuna-k8s/assets/15844821/66374cbd-f961-43ab-8b79-2daa80307e2b">

### 2. Filter single/multiple experiments
<img width="1439" alt="image" src="https://github.com/coderkol95/mlflow-optuna-k8s/assets/15844821/4b9000fc-478f-4b2c-9db7-813b185bfd53">

### 3. Filter runs
<img width="1440" alt="image" src="https://github.com/coderkol95/mlflow-optuna-k8s/assets/15844821/3726b5ab-b097-48d6-98a7-0b36826fe7d3">

### 4. Select models to register basis loss information
![image](https://github.com/coderkol95/mlflow-optuna-k8s/assets/15844821/a363adb7-a9b6-4ef5-84c1-d4cdd27cef98)

### 5. Enter their model names to register them
<img width="1440" alt="image" src="https://github.com/coderkol95/mlflow-optuna-k8s/assets/15844821/d692dbf4-98c8-48be-8953-295de326515a">

## Model registry
<img width="1440" alt="image" src="https://github.com/coderkol95/mlflow-optuna-k8s/assets/15844821/8e3a0707-bdad-46a2-88a7-ffe3b8ff7a10">


# Getting started

1. Clone the repo
2. Create a S3 bucket mlops-optuna with two folders inside: data/ and output/. Add X.csv and y.csv files inside data/. You will find them in data/ in this repository.
3. Add a .env file with access key and security key for a AWS user. Follow these steps:

 a. Go to your IAM in AWS and create a user with this policy:
```
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::mlops-optuna",
                "arn:aws:s3:::mlops-optuna/*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:PutObject"
            ],
            "Resource": [
                "arn:aws:s3:::mlops-optuna",
                "arn:aws:s3:::mlops-optuna/*"
            ]
        }
    ]
}
```
b. Generate Access key and Secret Access key from the Security credentials section
c. Add them to the .env file as below
```
AK="<access key>"
SK="<secret access key>"
```
5. Build a docker image from the Dockerfile with `docker build . -t mlops-webapp:3`

# Different options of running

## 1. Running the flask app locally
 a. Go the the root of the project directory and run `python src/train.py s3_trial 2 2 data 1e-5 1e-1 0.2 0.4 1 5` 
 b. Go to localhost:5001

## 2. Running through Docker locally
 a. `docker run -p 5001:5001 mlops-webapp:3` 
 b. Go to localhost:5001

## 3. Running through Kubernetes via Docker desktop

a. `kubectl apply -f k8s-deployment.yaml` -> This contains deployment, service and ingress manifests
b. Either do port forwarding `kubectl port-forward service/mlops-service 8080:80` or install ingress controller via `kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/cloud/deploy.yaml` and run `kubectl delete -A ValidatingWebhookConfiguration ingress-nginx-admission` followed by `kubectl get ingress`; then you will see the external IP.

# Cleanup of resources in case of K8s

`kubectl delete deploy,service,ingress -l  app=mlops`

# About the author

Anupam works at a leading pharmaceutical company as a ML engineer. He is passionate about bringing value to enterprises through production grade ML solutions. In his free time he loves listening to music, cooking, painting and going on trips. You may reach him at anupammisra1995@gmail.com.

# Future improvements

* Add option to enter custom ANN model architecture from UI
* Charts and more metrics as seen in mlFlow UI
* Better UI
* Security and resilience aspects if deployed to prod
