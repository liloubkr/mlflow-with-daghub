# API de prédiction : sert le modèle MLflow (RandomForest digits) sur le port 8080.
#   python train.py                 # produit ./model
#   docker build -t lbkr/mlflow-digits .
#   docker run -p 8080:8080 lbkr/mlflow-digits
#   POST http://localhost:8080/invocations  {"inputs": [[...64 pixels...]]}
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_DISABLE_AGENT_HINT=1

WORKDIR /app

# Dépendances d'abord : cette couche reste en cache tant que requirements.txt ne change pas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Le modèle exporté par train.py
COPY model/ ./model/

# Ne pas tourner en root
RUN useradd --create-home appuser
USER appuser

EXPOSE 8080
CMD ["mlflow", "models", "serve", "-m", "/app/model", "--env-manager", "local", "--host", "0.0.0.0", "--port", "8080"]
