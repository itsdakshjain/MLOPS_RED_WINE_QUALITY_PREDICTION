from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import pandas as pd
from mlProject.pipeline.prediction import PredictionPipeline
from pathlib import Path
import subprocess
import threading

app = Flask(__name__)

# This is our "is training running?" flag
training_lock = threading.Lock()
is_training = False
training_log = []

def run_training_in_background():
    """Run training in background so web app doesn't freeze"""
    global is_training, training_log
    training_log = []

    try:
        training_log.append("Training started...")
        result = subprocess.run(
            ["python", "main.py"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            training_log.append("Training completed successfully!")
            training_log.append(result.stdout)
        else:
            training_log.append("Training failed!")
            training_log.append(result.stderr or result.stdout)
    except Exception as e:
        training_log.append(f"Training error: {e}")
    finally:
        is_training = False  # reset the flag when done

def ensure_model_trained():
    """Auto-train model if it doesn't exist"""
    model_path = Path('artifacts/model_trainer/model.joblib')
    if not model_path.exists():
        print("Model not found. Starting automatic training...")
        try:
            os.system("python main.py")
            print("Auto-training completed!")
        except Exception as e:
            print(f"Auto-training failed: {e}")
    else:
        print("Model already exists, ready for predictions!")

@app.route('/', methods=['GET'])
def homePage():
    return render_template("index.html")

@app.route('/train', methods=['GET'])
def training():
    global is_training, training_log

    # If already training, don't start another one
    if is_training:
        return render_template(
            "train_status.html",
            training_success=None,
            training_log="Training is already in progress! Please wait..."
        )

    # Set the flag and start training in background
    is_training = True
    training_log = []
    thread = threading.Thread(target=run_training_in_background)
    thread.start()

    return render_template(
        "train_status.html",
        training_success=True,
        training_log="Training started in background! Check /train/status for updates."
    )

@app.route('/train/status', methods=['GET'])
def training_status():
    """Check if training is still running"""
    global is_training, training_log
    return jsonify({
        "is_training": is_training,
        "log": training_log
    })

@app.route('/predict', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        try:
            fixed_acidity = float(request.form['fixed_acidity'])
            volatile_acidity = float(request.form['volatile_acidity'])
            citric_acid = float(request.form['citric_acid'])
            residual_sugar = float(request.form['residual_sugar'])
            chlorides = float(request.form['chlorides'])
            free_sulfur_dioxide = float(request.form['free_sulfur_dioxide'])
            total_sulfur_dioxide = float(request.form['total_sulfur_dioxide'])
            density = float(request.form['density'])
            pH = float(request.form['pH'])
            sulphates = float(request.form['sulphates'])
            alcohol = float(request.form['alcohol'])

            data = [fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                    chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                    density, pH, sulphates, alcohol]
            data = np.array(data).reshape(1, 11)

            obj = PredictionPipeline()
            predict = obj.predict(data)
            return render_template('results.html', prediction=str(predict))
        except Exception as e:
            print('The Exception message is: ', e)
            return 'something is wrong'
    else:
        return render_template('index.html')

if __name__ == "__main__":
    print("Starting Wine Quality Prediction App...")
    ensure_model_trained()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)