from flask import Flask, request, jsonify

from src.model import fetch_model_predict

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


@app.route("/retrain", methods=["POST"])
def retrain():
    print("retrain")


@app.route("/train", methods=["POST"])
def train():
    print("train")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
