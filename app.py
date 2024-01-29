from flask import Flask,render_template,url_for,request
import mlflow
from src.analyze_runs import MLFlow_app_client
from utils.upload_to_s3 import upload_recursively_to_s3
import os
from datetime import datetime
import time
from dotenv import load_dotenv
load_dotenv()

####################################################################################################
# BACKEND SERVER LOGIC
####################################################################################################

app = Flask('mlops')
@app.errorhandler(404) 

def not_found(e):
    return render_template("404.html")

@app.route("/", methods=["GET","POST"])
def index():
    return render_template("index.html")

@app.route("/train", methods=["GET","POST"])
def train():
    if request.method=="POST":
        exp_name=request.form.get('name')
        input_path=request.form.get('input_path')
        artifact_path=request.form.get('artifact_path')
        epochs=request.form.get('epochs')
        trials=request.form.get('trials')
        lr_min=request.form.get('lr_min')
        lr_max=request.form.get('lr_max')
        dropout_min=request.form.get('dropout_min')
        dropout_max=request.form.get('dropout_max')
        n_layers_min=request.form.get('n_layers_min')
        n_layers_max=request.form.get('n_layers_max')

        try:
            run_experiment(exp_name,epochs,trials,input_path,artifact_path,lr_min,lr_max,dropout_min,dropout_max,n_layers_min,n_layers_max)
        except:
            raise RuntimeError("Error in training models")
        return render_template("index.html")
    return render_template("train.html")

@app.route("/experiments", methods=["GET","POST"])
def experiments():
    if request.method=="POST":
        if request.form.get('date-selector'):
            global date_selected
            date_selected=request.form.get('date-selector')
            filtered_exps=exp.dates_to_exps[date_selected]
            filtered_exp_names=exp.get_exp_names(filtered_exps)
            print(filtered_exp_names)
            return render_template("experiments.html", exps=filtered_exp_names)

        elif request.form.getlist('selected_exps'):
            exps_selected=request.form.getlist('selected_exps')
            print(exps_selected)
            experiment_ids=exp.experiment_names_to_ids(exps_selected)
            global run_names,filtered_runs
            filtered_runs,run_names=exp.get_run_names_in_exp(experiment_ids)
            return render_template("experiments.html", runs=run_names)
    
        elif request.form.getlist('selected_runs'):
            runs_selected=request.form.getlist('selected_runs')
            print(runs_selected)

            #get run ID

            selected_run_ids=[]

            for name,run_id in zip(run_names,filtered_runs):
                if name in runs_selected:
                    selected_run_ids.append(run_id)

            loss_table=exp.compare_losses(date_selected,selected_run_ids)
            loss_table_with_names=dict(zip(run_names,list(loss_table.values())))
            return render_template("experiments.html", losses=loss_table_with_names)
        
        elif request.form.getlist('selected_runs_to_register_model'):
            global runs_to_register_model
            runs_to_register_model=request.form.getlist('selected_runs_to_register_model')
            print(runs_to_register_model)

            runs_to_register_model_ids=exp.get_run_ids_from_names(runs_to_register_model)

            registered_models=exp.get_registered_models()
            dates_for_registered_models=exp.get_dates_of_runs(runs_to_register_model_ids)
            print(dates_for_registered_models)
            new_registration_details=[[x,y] for (x,y) in zip(dates_for_registered_models,runs_to_register_model_ids)]
            print(new_registration_details)
            return render_template("models.html",registered_models=registered_models, new_models=new_registration_details)

    return render_template("experiments.html",dates=exp.unique_dates)

@app.route("/models", methods=["GET","POST"])
def models():

    if request.form.getlist('model-name'):
        model_names=request.form.getlist('model-name')
        # model_versions=request.form.getlist('model-version')
        # model_stages=request.form.getlist('model-stage')
        registered_models=exp.get_registered_models()
        print(model_names)
        register_models(runs_to_register_model,model_names)
        return render_template("models.html",registered_models=registered_models)

    registered_models=exp.get_registered_models()
    return render_template("models.html",registered_models=registered_models)

@app.route("/endpoints", methods=["GET","POST"])
def endpoints():
    return render_template("endpoints.html")

####################################################################################################
# BACKEND TRAINING/REGISTRATION CALLS
####################################################################################################

def run_experiment(name, epochs, trials, input_path, artifact_path, lr_min, lr_max, dropout_min, dropout_max, n_layers_min, n_layers_max):

    mlflow.projects.run(
    uri=".",
    run_name="nn",
    entry_point="train",
    backend='local',
    synchronous=False,
    env_manager='local',
    parameters={
        'name':name,
        'epochs':epochs,
        'trials':trials,
        'input_path':input_path,
        'lr_min':lr_min,
        'lr_max':lr_max,
        'dropout_min':dropout_min,
        'dropout_max':dropout_max,
        'n_layers_min':n_layers_min,
        'n_layers_max':n_layers_max
    },
    )
    handle_artifact_upload(name, artifact_path)

def handle_artifact_upload(name, artifact_path):
    try:
        exp_end_time=datetime.fromtimestamp(mlfc.get_experiment_by_name(name=name).last_update_time/1e3)

        print("\n\n",exp_end_time)
        print(datetime.now())
        print("\n\n",)
        while (datetime.now()-exp_end_time).seconds/60<2:
            print((datetime.now()-exp_end_time)/2)
            print("Waiting...")
            time.sleep(30)
        upload_recursively_to_s3(artifact_path, AK, SK)
    except:
        print("Experiment not yet generated")

def register_models(runs_to_register_model,model_names):

    for run,model_name in zip(runs_to_register_model,model_names):
        mlflow.register_model(
            f"runs:/{run}/model", model_name,
            # version=version
            # stage=stage
            )

if __name__=="__main__":
    exp=MLFlow_app_client()
    mlfc=mlflow.MlflowClient()
    AK=os.environ["AK"]
    SK=os.environ["SK"]
    app.run(host="0.0.0.0", port=5001)
