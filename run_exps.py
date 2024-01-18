import mlflow

mlflow.projects.run(
    uri=".",
    run_name="e_30_lr_0.003",
    entry_point="train",
    backend='local', 
    synchronous=False,
    # parameters={
    #     'epochs': 3,
    #     'learning_rate':0.003
    # },
    )