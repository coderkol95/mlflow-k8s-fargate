import mlflow
from data_generator import generate_live_data
import mlflow.pyfunc
import numpy as np

model_name = "regression_NN_model"
# stage = 

model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/latest")

def predict_pipe(X):
    """
    Add more logic before prediction if required
    """
    predictions = model.predict(X)
    return predictions

if __name__=="__main__":

    X,y=generate_live_data(size=30)
    preds=predict_pipe(X.astype(np.float32))
    print(preds)