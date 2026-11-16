"""Vérifie que l'API du conteneur répond et prédit correctement.

    python tests/smoke_test.py http://localhost:8080

Attend que /ping réponde, envoie 50 images de test à /invocations
et échoue si l'accuracy est inférieure à 0.9.
"""
import json
import sys
import time
import urllib.request

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"
N_SAMPLES = 50
MIN_ACCURACY = 0.9


def wait_until_ready(timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if urllib.request.urlopen(f"{URL}/ping", timeout=2).status == 200:
                return
        except Exception:
            time.sleep(2)
    sys.exit(f"API not ready after {timeout}s")


def main():
    wait_until_ready()

    data = load_digits()
    _, X_test, _, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    X, y = X_test[:N_SAMPLES].tolist(), y_test[:N_SAMPLES].tolist()

    req = urllib.request.Request(
        f"{URL}/invocations",
        data=json.dumps({"inputs": X}).encode(),
        headers={"Content-Type": "application/json"},
    )
    predictions = json.loads(urllib.request.urlopen(req, timeout=30).read())["predictions"]

    accuracy = sum(p == t for p, t in zip(predictions, y)) / len(y)
    print(f"first predictions: {predictions[:10]}")
    print(f"true labels:       {y[:10]}")
    print(f"accuracy on {len(y)} samples: {accuracy:.2f}")
    if accuracy < MIN_ACCURACY:
        sys.exit(f"Accuracy {accuracy:.2f} < {MIN_ACCURACY}")
    print("Smoke test passed")


if __name__ == "__main__":
    main()
