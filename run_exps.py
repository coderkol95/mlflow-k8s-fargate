import mlflow

name='HP-Optuna'

mlflow.projects.run(
    uri=".",
    run_name=name,
    entry_point="train",
    backend='local', 
    synchronous=False,
    parameters={
        'name':name,
        'epochs': 3,
        'trials':2
    },
    )