import mlflow
import optuna
import lightning.pytorch as pl
import json
from datawork import data_module
from neural_network import NN
from datetime import datetime
from lightning.pytorch.callbacks import Callback
import argparse

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
    n_layers = trial.suggest_int("n_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    lr = trial.suggest_float("lr",1e-5,1e-1)

    output_dims = [
            trial.suggest_int(f"n_units_l{i}", 4, 128, log=True) for i in range(n_layers)
    ]
    od="_".join(str(x) for x in output_dims)
    version = f"version_{round(dropout,2)}_{round(lr,2)}_{od}"

    with mlflow.start_run(run_name=version, experiment_id=get_or_create_experiment(EXPERIMENT_NAME),nested=True) as run:
        
        mlflow.pytorch.autolog()
        mlflow.log_params(trial.params)
        pl.seed_everything(RANDOM_SEED, workers=True) # Setting seed for execution
        data=data_module(batch_size=4,seed=RANDOM_SEED)
        model = NN(dropout, output_dims,lr)

        trainer = pl.Trainer(
            logger=False,
            deterministic=True,
            enable_checkpointing=False,
            max_epochs=EPOCHS,
            default_root_dir="./",
        )
        trainer.fit(model,data)
        metrics=trainer.logged_metrics
        print("\n\n",metrics["train_loss"],"\n\n")
        data_to_log = {"date":str(datetime.today().date()),"runID": [run.info.run_id],"train_loss":metrics["train_loss"].numpy(),"val_loss":metrics["val_loss"].numpy()}
        print(data_to_log)
        mlflow.log_table(data=data_to_log, artifact_file="comparison_table.json")
        error = trainer.logged_metrics["val_loss"].item()
    
    return error

def get_or_create_experiment(experiment_name:str):

    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)

def train_model(experiment_id):
    # Initialize the Optuna study
    with mlflow.start_run(experiment_id=experiment_id,nested=True):
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=TRIALS) #, callbacks=[champion_callback])
        mlflow.log_params(study.best_params)
        mlflow.log_metric("val_loss",study.best_value)

if __name__=="__main__":

    a=argparse.ArgumentParser()
    a.add_argument("name", type=str)
    a.add_argument("epochs", type=int)
    a.add_argument("trials", type=int)

    args=a.parse_args()
    EXPERIMENT_NAME=args.name
    EPOCHS=args.epochs
    TRIALS=args.trials
    experiment_id = get_or_create_experiment(EXPERIMENT_NAME)
    train_model(experiment_id)