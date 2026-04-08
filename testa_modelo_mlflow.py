import mlflow
import mlflow.pyfunc
import pandas as pd

# Define o tracking URI para o servidor MLflow
mlflow.set_tracking_uri("http://localhost:5000")

# Agora carrega o modelo registrado
model = mlflow.pyfunc.load_model("models:/BridgeConditionRF/1")

# Carrega a amostra
sample = pd.read_csv("./output/sample_input.csv")

# Predição
predictions = model.predict(sample)
print(predictions)