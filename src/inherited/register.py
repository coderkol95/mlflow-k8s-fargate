import mlflow
import argparse
from datetime import datetime, timedelta
import pandas as pd

def register_model(model_name, metric,lookback_duration):

    table=mlflow.load_table(artifact_file='comparison_table.json')
    table=table[table['date'].notna()]
    start_date=pd.to_datetime('today').floor('D')-pd.Timedelta(days=int(lookback_duration))
    table['date']=table['date'].apply(lambda x:datetime.strptime(str(x),"%Y-%m-%d %H:%M:%S"))
    print(table.dtypes)
    table=table[table['date']>start_date]
    table.sort_values([metric], ascending=False, inplace=True)
    print(table.head())

    top_model_run_ID=table.head(1)['runID'].values[0]

    mlflow.register_model(
        f"runs:/{top_model_run_ID}/model", model_name,
        )

if __name__=="__main__":

    a=argparse.ArgumentParser()
    a.add_argument("--metric",type=str,default="test_loss")
    a.add_argument("--model_name", type=str, default=f"regression_NN_on{str(datetime.today().date()).replace('-','_')}")
    a.add_argument("--lookback_duration",type=str,default="7")
    args=a.parse_args()
    metric=args.metric
    model_name=args.model_name
    lookback_duration=args.lookback_duration

    register_model(model_name, metric,lookback_duration)