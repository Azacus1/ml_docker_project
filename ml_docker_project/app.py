from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")

target_names = ['setosa', 'versicolor', 'virginica']

@app.route("/predict", methods=["POST"])
def predict():
    """Make a prediction based on the input features.

    Example of input data:
    {
        "features": [1.0, 2.0, 3.0, 4.0]
    }

    :return: A JSON response with the prediction
    """
    data = request.get_json()
    if "features" not in data:
        return jsonify({"error": "Missing 'features' in request data."}), 400

    try:
        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)[0]
        response = {"prediction": target_names[prediction]}
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)