"""
Wine Quality Prediction - Flask Application
 
Security hardening (Issue 1 fix):
  - /train requires a secret token via TRAIN_SECRET env var (fail-closed if unset)
  - threading.Lock() replaces the bare boolean race condition
  - Flask-Limiter caps /train at 10 req/hour per IP, /train/status at 60/min
  - /train/status also requires the same token
  - debug mode is controlled by FLASK_DEBUG env var, defaults to off in production
"""
 
import os
import secrets
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
 
import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, render_template, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from pathlib import Path
from mlProject.config.configuration import ConfigurationManager
from mlProject.constants import ENV_FLASK_PORT, ENV_FLASK_DEBUG, ENV_TAG
from mlProject.pipeline.prediction import PredictionPipeline
from mlProject.utils.common import load_env_file, get_env_or_config
from mlProject.utils.model_registry import load_registry, rollback_to_version

load_env_file()
 
app = Flask(__name__)
 
# ---------------------------------------------------------------------------
# Rate limiter - keyed on caller's IP
# ---------------------------------------------------------------------------
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=[],          # No blanket global limit; set per-route
    storage_uri="memory://",    # Fine for single-worker Render free tier
)
 
# ---------------------------------------------------------------------------
# Training state
# _training_lock is held for the entire duration of a training run.
# acquire(blocking=False) is atomic - no race condition possible.
#
# Memory & thread safety:
# - training_log is a bounded deque(maxlen=100) to prevent unbounded growth
# - _log_lock protects all reads/writes to training_log
# - ThreadPoolExecutor(max_workers=1) replaces raw thread creation
# - Timeout mechanism auto-releases lock if training exceeds TRAIN_TIMEOUT
# ---------------------------------------------------------------------------
_training_lock = threading.Lock()
_log_lock = threading.Lock()
is_training = False
training_log = deque(maxlen=100)
_train_executor = ThreadPoolExecutor(max_workers=1)
TRAIN_TIMEOUT = int(os.environ.get("TRAIN_TIMEOUT", "1800"))  # default 30 min
 
 
# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------
def _verify_train_token() -> bool:
    """
    Validate the caller's secret token against the TRAIN_SECRET env var.
 
    Accepts the token in two ways (works from browser tab AND curl/CI):
      - Query param  : GET /train?token=<secret>
      - Custom header: X-Train-Token: <secret>
 
    Returns False (deny) if TRAIN_SECRET is not set in the environment -
    the endpoint is disabled entirely by default (fail-closed).
    Uses secrets.compare_digest to prevent timing-oracle attacks.
    """
    expected = os.environ.get("TRAIN_SECRET", "")
    if not expected:
        return False  # No secret configured - refuse everything
 
    supplied = (
        request.args.get("token", "")
        or request.headers.get("X-Train-Token", "")
    )
    return secrets.compare_digest(supplied.encode(), expected.encode())
 
 
# ---------------------------------------------------------------------------
# Startup helper
# ---------------------------------------------------------------------------
def validate_config_at_startup() -> None:
    """Validate all required configs are present before starting the server."""
    required_configs = [
        ("config/config.yaml", "Main configuration file"),
        ("params.yaml", "Parameters file"),
        ("schema.yaml", "Schema file"),
    ]
    missing = []
    for path, desc in required_configs:
        if not Path(path).exists():
            missing.append(f"{desc} ({path})")
    if missing:
        print(f"ERROR: Missing required configuration files: {', '.join(missing)}")
        sys.exit(1)
    print(f"Configuration validation passed. Environment: {os.environ.get(ENV_TAG, 'development')}")
 
 
# ---------------------------------------------------------------------------
# Background training worker
# ---------------------------------------------------------------------------
def _run_training_in_background() -> None:
    """Subprocess-based training; releases _training_lock when done."""
    global is_training
    start_time = time.time()
    try:
        with _log_lock:
            training_log.append("Training started...")
        result = subprocess.run(
            ["python", "main.py"],
            capture_output=True,
            text=True,
            timeout=TRAIN_TIMEOUT,
        )
        with _log_lock:
            if result.returncode == 0:
                training_log.append("Training completed successfully!")
                training_log.append(result.stdout)
            else:
                training_log.append("Training failed!")
                training_log.append(result.stderr or result.stdout)
    except subprocess.TimeoutExpired:
        with _log_lock:
            training_log.append(
                f"Training timed out after {TRAIN_TIMEOUT}s"
            )
    except Exception as exc:
        with _log_lock:
            training_log.append(f"Training error: {exc}")
    finally:
        is_training = False
        try:
            _training_lock.release()
        except RuntimeError:
            pass  # already released - should never happen
 
 
def ensure_model_trained() -> None:
    """Train the model automatically on first deploy if no artifact exists."""
    try:
        config_manager = ConfigurationManager()
        eval_config = config_manager.get_model_evaluation_config()
        model_path = eval_config.model_path
    except Exception:
        model_path = Path("artifacts/model_trainer/model.joblib")

    if not model_path.exists():
        print("Model not found - starting automatic training...")
        try:
            result = subprocess.run(
                ["python", "main.py"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print("Auto-training completed!")
            else:
                print(f"Auto-training failed:\n{result.stderr}")
        except Exception as exc:
            print(f"Auto-training failed: {exc}")
    else:
        from mlProject.utils.common import verify_model_integrity
        checksum_path = Path(str(model_path) + ".sha256")
        if not verify_model_integrity(model_path, checksum_path):
            print("Model integrity check FAILED - consider retraining.")
        else:
            print("Model already exists - ready for predictions!")
 
 
# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
 
@app.route("/", methods=["GET"])
def homePage():
    return render_template("index.html")
 
 
@app.route("/train", methods=["GET"])
@limiter.limit("10 per hour")           # Hard cap: 10 triggers/hr per IP
def training():
    global is_training

    # 1 - Authenticate first (401 reveals nothing about current state)
    if not _verify_train_token():
        abort(401)

    # 2 - Atomically acquire the lock; reject if already running
    acquired = _training_lock.acquire(blocking=False)
    if not acquired:
        return render_template(
            "train_status.html",
            training_success=None,
            training_log="Training is already in progress! Please wait...",
        )

    # 3 - Mark running and reset log buffer under lock
    is_training = True
    with _log_lock:
        training_log.clear()

    # 4 - Submit to ThreadPoolExecutor (reuses worker thread, avoids thread leak)
    _train_executor.submit(_run_training_in_background)

    return render_template(
        "train_status.html",
        training_success=True,
        training_log="Training started in background! Check /train/status for updates.",
    )
 
 
@app.route("/train/status", methods=["GET"])
@limiter.limit("60 per minute")         # Polling endpoint - more generous
def training_status():
    """Return current training state. Also requires the token."""
    if not _verify_train_token():
        abort(401)

    with _log_lock:
        log_snapshot = list(training_log)

    return jsonify({
        "is_training": is_training,
        "log": log_snapshot,
    })
 
 
@app.route("/predict", methods=["POST", "GET"])
def index():
    if request.method == "POST":
        try:
            fixed_acidity        = float(request.form["fixed_acidity"])
            volatile_acidity     = float(request.form["volatile_acidity"])
            citric_acid          = float(request.form["citric_acid"])
            residual_sugar       = float(request.form["residual_sugar"])
            chlorides            = float(request.form["chlorides"])
            free_sulfur_dioxide  = float(request.form["free_sulfur_dioxide"])
            total_sulfur_dioxide = float(request.form["total_sulfur_dioxide"])
            density              = float(request.form["density"])
            pH                   = float(request.form["pH"])
            sulphates            = float(request.form["sulphates"])
            alcohol              = float(request.form["alcohol"])
 
            data = np.array([
                fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                density, pH, sulphates, alcohol,
            ]).reshape(1, 11)
 
            obj = PredictionPipeline()
            predict = obj.predict(data)
            final_prediction = round(float(predict[0]), 2)
 
            return render_template("results.html", prediction=final_prediction)
 
        except Exception as exc:
            print("The Exception message is: ", exc)
            return render_template(
                "results.html",
                error_msg=(
                    "Unable to compute prediction. "
                    "Please ensure all fields are filled with valid numbers."
                ),
            )
    else:
        return render_template("index.html")


@app.route('/models', methods=['GET'])
def list_models():
    """List all registered model versions."""
    registry_path = Path('artifacts/model_registry.json')
    registry = load_registry(registry_path)
    return jsonify(registry)


@app.route('/models/compare', methods=['GET'])
def compare_models():
    """Show metric diff between current and previous model."""
    comparison_path = Path('artifacts/model_evaluation/metrics_comparison.json')
    if comparison_path.exists():
        try:
            with open(comparison_path) as f:
                import json
                comparison = json.load(f)
            return jsonify(comparison)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"message": "No comparison data available"})


@app.route('/models/rollback', methods=['POST'])
def rollback_model():
    """Rollback production alias to a specified version."""
    version_id = request.json.get("version_id")
    if not version_id:
        return jsonify({"error": "version_id is required"}), 400
    registry_path = Path('artifacts/model_registry.json')
    if rollback_to_version(registry_path, version_id):
        return jsonify({"message": f"Rolled back to version {version_id}"})
    return jsonify({"error": f"Version {version_id} not found"}), 404


@app.route('/models/<version_id>', methods=['GET'])
def get_model_version(version_id):
    """View metadata for a specific version."""
    registry_path = Path('artifacts/model_registry.json')
    registry = load_registry(registry_path)
    for v in registry.get("versions", []):
        if v.get("id") == version_id:
            return jsonify(v)
    return jsonify({"error": f"Version {version_id} not found"}), 404


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Starting Wine Quality Prediction App...")
    validate_config_at_startup()
    ensure_model_trained()
 
    # debug=True in production exposes an interactive shell - never do this.
    # Set FLASK_DEBUG=1 locally to enable the Werkzeug debugger.
    port = int(get_env_or_config(ENV_FLASK_PORT, "8080", transform=int))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
 
    app.run(host="0.0.0.0", port=port, debug=debug)
 