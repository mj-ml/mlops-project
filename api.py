from flask import Flask, request, jsonify

from src.model import fetch_model_predict, register_the_best_model
from src.monitoring import monitoring
from src.pipeline import training_pipeline

app = Flask("load-prediction")


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    ride = request.get_json()
    pred = fetch_model_predict(
        **ride,
    )[0]
    result = {"load": pred}
    return jsonify(result)


@app.route("/alive", methods=["POST"])
def alive():
    print("alive")
    return jsonify({"status": "ok"})


@app.route("/monitoring", methods=["POST"])
def retrain():
    print("monitoring")
    monitoring()
    return jsonify({"status": "ok"})


@app.route("/train", methods=["POST"])
def train():
    print("train")
    training_pipeline()
    register_the_best_model(top_n=2)
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
