# MLOps platform running on Kubernetes

## Introduction

This project aims to be a MLOPS template for UI friendly deep learning development using Optuna and Pytorch running on a K8s cluster.

The following features are demonstrated:
* Download preprocessed training data from AWS S3 path given by user
* Train an artificial neural network using [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/).
* Automatic hyperparameter tuning using [Optuna](https://optuna.org/).
* Lifecycle management of the project using [mlFlow](https://mlflow.org/).
* Provide a web interface built using Flask, HTML and CSS.
* Container execution in Kubernetes cluster

## Advantages of this approach

* Simple UI based controls for model development, experiment analysis and model registration.
* Dockerization ensures portability, repeatability and scalability.
* Optuna automatically finds the best hyperparameters [Hyperparameter ranges are supplied via the UI].
* Automatic resilience, scaling and load balancing via Kubernetes.
* Stateful data fetching and experiment run saving from/to S3.

```
This template can be extended and much much more features can be added!!!
```

# Directory structure

```
/
 └──.github/
      └── workflows/                    
             push_app_image_to_ecr.yaml      # Pushes the updated web application image to AWS ECR using Github Actions; Also updates the K8s manifest to use the updated image
 └── data/                    
         X.csv                               # Training input features file
         y.csv                               # Training input target file
 └── {s3.folder}/ {s3.input files}           # Training data is downloaded from s3 folder given by user to local folder with same name
 └── mlruns/
      └── {experiment_ID}/
             └── {run_ID}/ {run artifacts}   # Run artifacts are stored here
 └── src/
         train.py                            # MLProject uses this file for model training
         neural_network.py                   # The neural network module implemented using lightning.pytorch.LightningModule
         datawork.py                         # The data preprocessing module implemented using lightning.pytorch.LightningDataModule
         analyze_runs.py                     # The module which provides required data to the web app from the MLFlowClient
 └── static/
     └── css/                                # CSS files
             index.css                       
             train.css
             experiments.css
             models.css
             404.css
 └── templates/                              # HTML files
             index.html
             train.html
             experiments.html
             models.html
             404.html
 └── utils/
         upload_to_s3.py                     # This file recursively uploads the mlruns/ after each training experiment
 .env                                        # Environment variables are stored here AK and SK
 Dockerfile                                  # Simple containerisation and exposing the Flask web app on port 5001
 .gitignore                                  # Ignoring unnecesary files
 app.py                                      # The Flask web application
 config.json                                 # Random state value, batch size and other configurations for lightning.pytorch are fetched from here
 k8s-deployment.yaml                         # The Kubernetes deployments, service and ingress details
 LICENSE                                     # MIT license
 MLProject                                   # MLProject file used by the web application. MLProject is run using an environment as it is already running inside a dockerised web app
 python_env.yaml                             # The environment creation details for running mlFLow experiment
 requirements.txt                            # Requirements file for python environment

```

# Links between different files

## Dockerfile

```
EXPOSE 5001 -> The port exposed in the docker container. This is used in k8s deployment containerPort
```

## K8s-deployment.yaml

```
5. labels:
     app: mlops -> Used everywhere for identifying the app. I have not used namespace as I am running only one K8s application.
19. image: mlops-webapp:3 -> The docker image and version. If you are rebuilding the image, ensure that the correct image is used
21. containerPort: 5001 -> The container port mentioned in the Dockerfile
39. port: 80 -> The port on which this web app service is running
    targetPort: 5001 -> The containerPort of the web app
63. name: mlops-service
      port:
        number: 80 -> The port of the service mentioned above
```

## MLProject

```
3. python_env: python_env.yaml -> As the web app is already running in a container, using the environment creation route for running the MLFlow experiments. This is done to keep the deployment and application simple instead of using nested dockers.
```

## src/train.py

```
All the hyper parameters from the UI are used here. If you add any HP, use the below route to ensure the inputs flow correctly.

Hyperparameter route:

train.html & css.html -> app.py train() ->app.py run_experiment() -> MLProject train: -> src/train.py __main__ 

```

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

# Application snapshots

## Landing page
<img width="1439" alt="image" src="https://github.com/coderkol95/mlflow-optuna-k8s/assets/15844821/a26e3888-9908-42f4-b7b2-ecf9a8007e1e">

## Model training page
<img width="1438" alt="image" src="https://github.com/coderkol95/mlflow-optuna-k8s/assets/15844821/f0f54221-6685-4c35-b558-3cc64e8e450f">

## Experiment analysis journey
### 1. Select the date
<img width="1440" alt="image" src="https://github.com/coderkol95/mlflow-optuna-k8s/assets/15844821/77b1888e-af43-4391-90ba-d2a5db37caab">

### 2. Filter single/multiple experiments
<img width="1439" alt="image" src="https://github.com/coderkol95/mlflow-optuna-k8s/assets/15844821/4b9000fc-478f-4b2c-9db7-813b185bfd53">

### 3. Filter runs
<img width="1440" alt="image" src="https://github.com/coderkol95/mlflow-optuna-k8s/assets/15844821/018ed758-c2cd-42ed-bad7-6204ee977bcb">

### 4. Select models to register basis loss information
![image](https://github.com/coderkol95/mlflow-optuna-k8s/assets/15844821/a363adb7-a9b6-4ef5-84c1-d4cdd27cef98)

### 5. Enter their model names to register them
<img width="1440" alt="image" src="https://github.com/coderkol95/mlflow-optuna-k8s/assets/15844821/d692dbf4-98c8-48be-8953-295de326515a">

## Model registry
<img width="1440" alt="image" src="https://github.com/coderkol95/mlflow-optuna-k8s/assets/15844821/8e3a0707-bdad-46a2-88a7-ffe3b8ff7a10">

## Runs details stored in s3
<img width="1440" alt="image" src="https://github.com/coderkol95/mlflow-optuna-k8s/assets/15844821/96ffca7d-f23e-4357-94c8-07962b6a0a3f">

## Image uploaded to ECR
<img width="1440" alt="image" src="https://github.com/coderkol95/mlflow-optuna-k8s/assets/15844821/a3b11923-e8e0-4c26-903f-e496f8a3cfb2">

# Future improvements

* Add option to enter custom ANN model architecture from UI
* Charts and more metrics as seen in mlFlow UI
* Better UI
* Security and resilience aspects if deployed to prod
