"""Entraîne le RandomForest du TP MLflow et l'exporte au format MLflow dans ./model.

Utilisé par la CI (.github/workflows/cicd.yml) avant de construire l'image Docker :
    python train.py
Le run est enregistré dans MLflow (paramètres, métrique, modèle), puis le modèle est
copié dans ./model pour être embarqué dans l'image. Si l'accuracy est trop basse,
le script échoue et rien n'est construit ni poussé.
"""
import os
import shutil
import sys

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

N_ESTIMATORS = 100
MAX_DEPTH = 10
MIN_ACCURACY = 0.95  # garde-fou : en dessous, on ne publie pas le modèle
MODEL_DIR = "model"


def main():
    data = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    mlflow.set_experiment("RandomForest_CI")
    run_name = "ci-" + os.environ.get("GITHUB_SHA", "local")[:7]

    with mlflow.start_run(run_name=run_name):
        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=42
        )
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))

        mlflow.log_param("n_estimators", N_ESTIMATORS)
        mlflow.log_param("max_depth", MAX_DEPTH)
        mlflow.log_metric("accuracy", accuracy)

        # Export au format MLflow (MLmodel, requirements, modèle skops) dans ./model
        shutil.rmtree(MODEL_DIR, ignore_errors=True)
        mlflow.sklearn.save_model(
            model,
            MODEL_DIR,
            input_example=X_train[:5],
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        mlflow.log_artifacts(MODEL_DIR, artifact_path="model")

    print(f"accuracy={accuracy:.4f} (minimum {MIN_ACCURACY})")
    if accuracy < MIN_ACCURACY:
        sys.exit(f"Accuracy {accuracy:.4f} < {MIN_ACCURACY}: model not published")
    print(f"Model exported to ./{MODEL_DIR}")


if __name__ == "__main__":
    main()
