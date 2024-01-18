import lightning.pytorch as pl
from datawork import data_module
from neural_network import NN
import os
import optuna
# from optuna.integration import PyTorchLightningPruningCallback # This is causing a lot of problems between packages lightning and pytorch-lightning. Removing it for now
from typing import List
import json
import argparse

# When executing locally: python code/train.py --local True

# Pytorch 2 has problem with last linear layer having 1 cell in arm arch. Hence reverted to prev version
# Optuna does not work with pytorch lightning >=2.0, using 1.8
EPOCHS:int=0

with open("config.json","r") as f:
    configs=json.load(f)

RANDOM_SEED:int=configs["RANDOM STATE"]

def objective(trial):

    # We optimize the number of layers, hidden units in each layer, dropout and the learning rate.
    n_layers = trial.suggest_int("n_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    lr = trial.suggest_float("learning_rate",1e-5,1e-1)

    output_dims = [
        trial.suggest_int(f"n_units_l{i}", 4, 128, log=True) for i in range(n_layers)
    ]

    od="_".join(str(x) for x in output_dims)

    # version = f"version_{round(dropout,2)}_{round(lr,2)}_{od}"

    pl.seed_everything(RANDOM_SEED, workers=True) # Setting seed for execution
    model = NN(dropout, output_dims,lr)

    trainer = pl.Trainer(
        logger=True,
        deterministic=True,
        # limit_val_batches=PERCENT_VALID_EXAMPLES,
        enable_checkpointing=False,
        max_epochs=EPOCHS,
        # callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")],
        default_root_dir=output_path
    )
    hyperparameters = dict(n_layers=n_layers, dropout=dropout, output_dims=output_dims, lr=lr)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model,data)

    return trainer.callback_metrics["val_loss"].item()

if __name__ =='__main__':

    try:
        # Hyperparameters received when run as Sagemaker image
        hyperparams = json.load(open(hyperparam_path))        
        EPOCHS=int(hyperparams["epochs"]) if "epochs" in list(hyperparams.keys()) else 3
        # Receive other hyperparams maybe?
    except:
        EPOCHS=2

    a = argparse.ArgumentParser()
    a.add_argument("--local",default=False, type=bool, required=False)

    parsed_args=a.parse_args()

    if parsed_args.local:
        data=data_module(folder="opt/ml/input/data/backup/")
    else:
        data=data_module()

    pruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=2, timeout=300)

    print("Best trial:")
    trial = study.best_trial

    print(f"  Validation loss: {trial.value}")

    print("  Best model's parameters: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    json.dump(trial.params,open('best_trial_params.json','w'))

# Not saving every model. It will take up a lot of space, resulting in a lot of unnecessary cost. 
# Instead enforcing seed and deterministic run of Trainer 