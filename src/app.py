from flask import Flask, request, jsonify
import mlflow.pyfunc
import pandas as pd

app = Flask(__name__)

# Carrega o modelo registrado no MLflow (estágio Production)
model = mlflow.pyfunc.load_model("models:/BridgeConditionRF/Production")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame(data)
        preds = model.predict(df)
        probas = model.predict_proba(df) if hasattr(model, 'predict_proba') else None
        return jsonify({
            'predictions': preds.tolist(),
            'probabilities': probas.tolist() if probas is not None else None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)