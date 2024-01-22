import mlflow
from glob import glob
import datetime
import json

class get_past_experiments_details():
    def __init__(self):
        self.client=mlflow.MlflowClient()
        self.generate_dates_to_exps_mappings()
        self.generate_exps_to_runs_mappings()

    def get_dates_of_runs(self,runs):
        timestamps=[]
        for run in runs:
            run_end_time=str(datetime.datetime.fromtimestamp(self.client.get_run(run).info.end_time/1e3))
            timestamps.append(run_end_time)
        return timestamps

    def get_registered_models(self):
        register_model_details=[]
        for model in self.client.search_registered_models():
            name=model.name
            run_id=model.latest_versions[0].run_id
            # stage=model.latest_versions[0].current_stage
            date_updated=str(datetime.datetime.fromtimestamp(model.latest_versions[0].last_updated_timestamp/1e3)).split(' ')[0]
            register_model_details.append([name,date_updated,run_id])
        return register_model_details

    def generate_dates_to_exps_mappings(self):
        self.exp_ids=[[x.experiment_id,str(datetime.datetime.fromtimestamp(x.last_update_time/1e3)).split(' ')[0]] for x in self.client.search_experiments()][:-1] # removing the default exp
        self.unique_dates=list(set([i[1] for i in self.exp_ids]))
        self.dates_to_exps={d:[] for d in self.unique_dates}
        for d in self.dates_to_exps:
            for i in self.exp_ids:
                if d in i:
                    self.dates_to_exps[d].append(i[0])
        self.experiment_ids=[ e for es in self.dates_to_exps.values() for e in es]

    def generate_exps_to_runs_mappings(self):
        self.exps_to_runs={y:[x.split('/')[-1] for x in glob(f"mlruns/{y}/*") if 'meta' not in x] for y in self.experiment_ids}
    
    def get_exp_and_run_losses_for_date_detailed(self,date:str):

        exps=self.dates_to_exps[date]

        losses_on_date={}
        for exp in exps:
            
            losses_on_date[exp]={}
            runs=self.exps_to_runs[exp]
            for run in runs:
                run_losses=self.get_run_losses_detailed(exp,run)
                losses_on_date[exp][run]={}
                losses_on_date[exp][run]=run_losses

        return losses_on_date
    
    def get_exp_and_run_losses_for_date_table(self,date:str):

        exps=self.dates_to_exps[date]

        losses_on_date={}
        for exp in exps:
            
            losses_on_date[exp]={}
            runs=self.exps_to_runs[exp]
            for run in runs:
                run_losses=self.get_run_losses_table(exp,run)
                losses_on_date[exp][run]={}
                losses_on_date[exp][run]=run_losses

        return losses_on_date

    def get_run_losses_detailed(self,exp_id,run_id):
        x=[]
        with open(f'mlruns/{exp_id}/{run_id}/metrics/train_loss') as f:
            x=f.read()
            train_losses={y[2]:y[1] for y in [y.split(' ') for y in x.split('\n')][:-1]}
        with open(f'mlruns/{exp_id}/{run_id}/metrics/val_loss') as f:
            x=f.read()
            val_losses={y[2]:y[1] for y in [y.split(' ') for y in x.split('\n')][:-1]}
        epochs=list(train_losses.keys())
        losses={}
        for e in epochs:
            losses[e]=[train_losses[e],val_losses[e]]
        return losses
    
    def get_run_losses_table(self,exp_id,run_id):
        x=[]
        with open(f'mlruns/{exp_id}/{run_id}/artifacts/comparison_table.json','r') as f:
            json_file=json.load(f)
            return [json_file["data"][0][2],json_file["data"][0][3]]
    
    def compare_losses(self,date,runs):
        losses_table=self.get_exp_and_run_losses_for_date_table(date)
        run_losses=dict(x for row in losses_table.values() for x in row.items())
        return {r:run_losses[r] for r in runs}

    def get_run_ids(self,exps):

        runs=[]
        for e in exps:
            runs.extend(self.exps_to_runs[e])
        return runs
    
    def get_exp_names(self,exp_ids):

        names=[]
        for exp_id in exp_ids:
            names.append(self.client.get_experiment(exp_id).name)
        return names
    
    def experiment_names_to_ids(self, names):

        exp_ids=[]
        for name in names:
            exp_ids.append(self.client.get_experiment_by_name(name).experiment_id)
        return exp_ids