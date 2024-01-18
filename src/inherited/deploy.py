from mlflow import MlflowClient
import argparse

client = MlflowClient()


def update_model_state(model_name,model_version,model_stage):

    client.transition_model_version_stage(
        name=model_name, version=model_version, stage=model_stage
    )

if __name__=="__main__":

    a=argparse.ArgumentParser()
    a.add_argument("--model_name", type=str, default="regression_NN_model")
    a.add_argument("--model_version", type=str, default="latest")
    a.add_argument("--model_stage", type=str, default="Staging")
    args=a.parse_args()
    model_name=args.model_name
    model_version=args.model_version
    model_stage=args.model_stage

    update_model_state(model_name,model_version,model_stage)