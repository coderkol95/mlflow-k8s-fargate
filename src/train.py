import mlflow
import optuna
import lightning.pytorch as pl
import json
from datawork import data_module
from neural_network import NN
from datetime import datetime
from lightning.pytorch.callbacks import Callback
import argparse
import os
from dotenv import load_dotenv
load_dotenv()

# override Optuna's default logging to ERROR only
optuna.logging.set_verbosity(optuna.logging.ERROR)

with open("config.json","r") as f:
    configs=json.load(f)

RANDOM_SEED=configs["RANDOM STATE"]

class log_losses(Callback):

    def on_train_epoch_end(self, trainer, pl_module):
        mlflow.log_metric('train_loss_epochs', trainer.logged_metrics['train_loss'])
    def on_validation_epoch_end(self, trainer, pl_module):
        mlflow.log_metric('val_loss_epochs', trainer.logged_metrics['val_loss'])

def objective(trial):
    # We optimize the number of layers, hidden units in each layer, dropout and the learning rate.
    n_layers = trial.suggest_int("n_layers", N_LAYERS_MIN, N_LAYERS_MAX)
    dropout = trial.suggest_float("dropout", DROPOUT_MIN, DROPOUT_MAX)
    lr = trial.suggest_float("lr",LR_MIN,LR_MAX)

    output_dims = [
            trial.suggest_int(f"n_units_l{i}", 4, 128, log=True) for i in range(n_layers)
    ]
    od="_".join(str(x) for x in output_dims)
    version = f"version_{round(dropout,2)}_{round(lr,2)}_{od}"

    with mlflow.start_run(run_name=version, experiment_id=EXPERIMENT_ID,nested=True) as run:
        
        mlflow.pytorch.autolog()
        mlflow.log_params(trial.params)
        pl.seed_everything(RANDOM_SEED, workers=True) # Setting seed for execution
        data=data_module(4,RANDOM_SEED, INPUT_PATH,AK,SK)
        model = NN(dropout, output_dims,lr)

        trainer = pl.Trainer(
            logger=False,
            deterministic=True,
            enable_checkpointing=False,
            max_epochs=EPOCHS,
            default_root_dir="./"
        )
        trainer.fit(model,data)
        metrics=trainer.logged_metrics
        print("\n\n",metrics["train_loss"],"\n\n")
        data_to_log = {"date":str(datetime.today().date()),"runID": [run.info.run_id],"train_loss":metrics["train_loss"].numpy(),"val_loss":metrics["val_loss"].numpy()}
        print(data_to_log)
        mlflow.log_table(data=data_to_log, artifact_file="comparison_table.json")
        error = trainer.logged_metrics["val_loss"].item()
    
    return error

def train_model():
    # Initialize the Optuna study
    with mlflow.start_run(experiment_id=EXPERIMENT_ID, nested=True):
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=TRIALS) #, callbacks=[champion_callback])
        mlflow.log_params(study.best_params)
        mlflow.log_metric("val_loss",study.best_value)

def get_or_create_experiment():

    if experiment := mlflow.get_experiment_by_name(EXPERIMENT_NAME):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(EXPERIMENT_NAME,artifact_location=ARTIFACT_PATH)

def args_handler():

    a=argparse.ArgumentParser()
    a.add_argument("name", type=str)
    a.add_argument("epochs", type=int)
    a.add_argument("trials", type=int)
    a.add_argument("input_path", type=str)
    a.add_argument("artifact_path", type=str)
    a.add_argument("lr_min", type=float)
    a.add_argument("lr_max", type=float)
    a.add_argument("dropout_min", type=float)
    a.add_argument("dropout_max", type=float)
    a.add_argument("n_layers_min", type=int)
    a.add_argument("n_layers_max", type=int)
    args=a.parse_args()
    return args

if __name__=="__main__":

    args=args_handler()

    EXPERIMENT_NAME=args.name
    EPOCHS=args.epochs
    TRIALS=args.trials

    if EXPERIMENT_NAME=="local":
        INPUT_PATH='data'
    else:
        INPUT_PATH=args.input_path
    ARTIFACT_PATH=args.artifact_path

    LR_MIN=args.lr_min
    LR_MAX=args.lr_max

    N_LAYERS_MIN=args.n_layers_min
    N_LAYERS_MAX=args.n_layers_max

    DROPOUT_MIN=args.dropout_min
    DROPOUT_MAX=args.dropout_max

    AK=os.environ["AK"]
    SK=os.environ["SK"]

    EXPERIMENT_ID = get_or_create_experiment()
    print("\n\n",EXPERIMENT_ID,"\n\n")
    train_model()