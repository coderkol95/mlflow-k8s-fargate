from flask import Flask,render_template,url_for,request
import mlflow
from src.analyze_runs import get_past_experiments_details

####################################################################################################
# BACKEND SERVER LOGIC
####################################################################################################

app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
def index():
    return render_template("index.html")

@app.route("/train", methods=["GET","POST"])
def train():
    if request.method=="POST":
        exp_name=request.form.get('name')
        epochs=request.form.get('epochs')
        trials=request.form.get('trials')
        try:
            run_experiment(exp_name,epochs,trials)
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
            filtered_runs=exp.get_run_ids(experiment_ids)
            return render_template("experiments.html", runs=filtered_runs)
    
        elif request.form.getlist('selected_runs'):
            runs_selected=request.form.getlist('selected_runs')
            print(runs_selected)
            loss_table=exp.compare_losses(date_selected,runs_selected)

            return render_template("experiments.html", losses=loss_table)
        
        elif request.form.getlist('selected_runs_to_register_model'):
            global runs_to_register_model
            runs_to_register_model=request.form.getlist('selected_runs_to_register_model')
            print(runs_to_register_model)

            registered_models=exp.get_registered_models()
            dates_for_registered_models=exp.get_dates_of_runs(runs_to_register_model)
            print(dates_for_registered_models)
            new_registration_details=[[x,y] for (x,y) in zip(dates_for_registered_models,runs_to_register_model)]
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
# BACKEND TRAINING/REGISTRATION CALL
####################################################################################################

def run_experiment(name, epochs, trials):

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
        'trials':trials
    },
    )

def register_models(runs_to_register_model,model_names):

    
    for run,model_name in zip(runs_to_register_model,model_names):
        mlflow.register_model(
            f"runs:/{run}/model", model_name,
            # version=version
            # stage=stage
            )

if __name__=="__main__":
    exp=get_past_experiments_details()
    app.run(debug=True, port=5001)
