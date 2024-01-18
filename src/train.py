import mlflow
import optuna
import lightning.pytorch as pl
import json
from src.datawork import data_module
from src.neural_network import NN
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import MLFlowLogger

# override Optuna's default logging to ERROR only
optuna.logging.set_verbosity(optuna.logging.ERROR)

with open("config.json","r") as f:
    configs=json.load(f)

RANDOM_SEED:int=configs["RANDOM STATE"]
EPOCHS:int=configs["EPOCHS"]
TRIALS:int=configs["TRIALS"]
EXPERIMENT_NAME="Nested false"

def objective(trial):

    with mlflow.start_run(nested=True):

        # We optimize the number of layers, hidden units in each layer, dropout and the learning rate.
        n_layers = trial.suggest_int("n_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.2, 0.5)
        lr = trial.suggest_float("learning_rate",1e-5,1e-1)

        output_dims = [
            trial.suggest_int(f"n_units_l{i}", 4, 128, log=True) for i in range(n_layers)
        ]

        # od="_".join(str(x) for x in output_dims)
        # version = f"version_{round(dropout,2)}_{round(lr,2)}_{od}"

        pl.seed_everything(RANDOM_SEED, workers=True) # Setting seed for execution
        data=data_module(batch_size=4,seed=RANDOM_SEED)
        model = NN(dropout, output_dims,lr)

        mlf_logger = MLFlowLogger(experiment_name=EXPERIMENT_NAME) #, tracking_uri="file:./ml-runs")

        trainer = pl.Trainer(
            logger=mlf_logger,
            deterministic=True,
            enable_checkpointing=False,
            max_epochs=EPOCHS,
            default_root_dir="./"
        )
        hyperparameters = dict(n_layers=n_layers, dropout=dropout, output_dims=output_dims, lr=lr)
        trainer.fit(model,data)
        error = trainer.callback_metrics["val_loss"].item()

    return error

def get_or_create_experiment(experiment_name:str):

    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)

def train_model():
    with mlflow.start_run(experiment_id=experiment_id, run_name="Experiment run", nested=True):
        # Initialize the Optuna study
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=TRIALS) #, callbacks=[champion_callback])

if __name__=="__main__":
    experiment_id = get_or_create_experiment(EXPERIMENT_NAME)
    experiment_id

    # Set the current active MLflow experiment
    mlflow.set_experiment(experiment_id=experiment_id)

    train_model()