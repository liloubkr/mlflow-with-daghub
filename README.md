# mlflow-with-daghub

TP *ML in prod* : versioning des données avec **DVC**, suivi des expériences avec **MLflow** (remote **DagsHub**), puis déploiement du modèle en API via **Docker** et **GitHub Actions**.

| Élément | Où |
|---|---|
| Données versionnées (DVC, remote Google Drive) | `data/raw_dataset.csv.dvc` |
| Notebook MLflow : expériences, registry, DagsHub | `MLFlowSetup.ipynb` |
| Expériences MLflow | https://dagshub.com/liloubkr/mlflow-with-daghub.mlflow |
| Entraînement + export du modèle | `train.py` |
| Image de l'API | `Dockerfile` |
| Pipeline CI/CD | `.github/workflows/cicd.yml` |
| Image Docker Hub | https://hub.docker.com/r/lbkr/mlflow-digits |

## Pipeline CI/CD

À chaque push sur `main` :

1. `python train.py` entraîne un RandomForest sur le dataset *digits*, enregistre le run dans MLflow et exporte le modèle dans `./model`. Le pipeline échoue si l'accuracy est inférieure à 0.95.
2. `docker build` crée une image qui sert le modèle avec `mlflow models serve` sur le port 8080.
3. Le conteneur est démarré, puis `tests/smoke_test.py` lui envoie 50 images de test et vérifie les prédictions.
4. Si tout passe, l'image est poussée sur Docker Hub avec les tags `latest` et `<sha du commit>`.

## Lancer l'API

```bash
docker run -p 8080:8080 lbkr/mlflow-digits:latest
```

Exemple de requête (le fichier d'exemple est créé par `python train.py`) :

```bash
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d @model/serving_input_example.json
```

Le corps attendu est `{"inputs": [[64 pixels], ...]}` et la réponse est `{"predictions": [...]}`.
