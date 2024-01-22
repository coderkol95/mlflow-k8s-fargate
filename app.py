from flask import Flask,render_template,url_for,request
import mlflow

####################################################################################################
# BACKEND SERVER LOGIC
####################################################################################################

app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
def index():
    return render_template("index.html")

@app.route("/train", methods=["GET","POST"])
def train():

    if request.method=="POST":

        exp_name=request.form.get('name')
        epochs=request.form.get('epochs')
        trials=request.form.get('trials')
        
        try:
            run_experiment(exp_name,epochs,trials)
        except:
            raise RuntimeError("Error in training models")

        return render_template("index.html")
    return render_template("train.html")

@app.route("/experiments", methods=["GET","POST"])
def experiments():
    return render_template("experiments.html")

@app.route("/models", methods=["GET","POST"])
def models():
    return render_template("models.html")

@app.route("/endpoints", methods=["GET","POST"])
def endpoints():
    return render_template("endpoints.html")

####################################################################################################
# BACKEND TRAINING LOGIC
####################################################################################################

def run_experiment(name, epochs, trials):

    mlflow.projects.run(
    uri=".",
    run_name=name,
    entry_point="train",
    backend='local', 
    synchronous=False,
    parameters={
        'name':name,
        'epochs':epochs,
        'trials':trials
    },
    )

if __name__=="__main__":
    app.run(debug=True, port=5001)