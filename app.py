from flask import Flask, request, jsonify
import mlflow.pyfunc
import pandas as pd

app = Flask(__name__)
model = mlflow.pyfunc.load_model("models:/BridgeConditionRF/Production")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame(data)
    preds = model.predict(df)
    return jsonify({'predictions': preds.tolist()})

if __name__ == '__main__':
    app.run(port=5001)