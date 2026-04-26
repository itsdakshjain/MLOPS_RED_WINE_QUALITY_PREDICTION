from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from mlProject.pipeline.prediction import PredictionPipeline
from pathlib import Path
import subprocess

# Testing CI/CD pipeline
app = Flask(__name__) # initializing a flask app

# def ensure_model_trained():
#     """Auto-train model if it doesn't exist or is outdated"""
#     model_path = Path('artifacts/model_trainer/model.joblib')
    
#     if not model_path.exists():
#         print("🔄 Model not found. Starting automatic training...")
#         try:
#             os.system("python main.py")
#             print("✅ Auto-training completed successfully!")
#         except Exception as e:
#             print(f"❌ Auto-training failed: {e}")
#     else:
#         print("✅ Model already exists, ready for predictions!")

# # Auto-train on startup
# print("🚀 Starting Wine Quality Prediction App...")
# ensure_model_trained()


@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")


@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    try:
        result = subprocess.run(["python", "main.py"], capture_output=True, text=True)
        training_success = result.returncode == 0
        training_log = result.stdout if training_success else result.stderr or result.stdout

        return render_template(
            "train_status.html",
            training_success=training_success,
            training_log=training_log,
        )
    except Exception as e:
        return render_template(
            "train_status.html",
            training_success=False,
            training_log=str(e),
        )


@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            fixed_acidity =float(request.form['fixed_acidity'])
            volatile_acidity =float(request.form['volatile_acidity'])
            citric_acid =float(request.form['citric_acid'])
            residual_sugar =float(request.form['residual_sugar'])
            chlorides =float(request.form['chlorides'])
            free_sulfur_dioxide =float(request.form['free_sulfur_dioxide'])
            total_sulfur_dioxide =float(request.form['total_sulfur_dioxide'])
            density =float(request.form['density'])
            pH =float(request.form['pH'])
            sulphates =float(request.form['sulphates'])
            alcohol =float(request.form['alcohol'])
       
         
            data = [fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol]
            data = np.array(data).reshape(1, 11)
            
            obj = PredictionPipeline()
            predict = obj.predict(data)

            return render_template('results.html', prediction = str(predict))

        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'

    else:
        return render_template('index.html')
    


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)