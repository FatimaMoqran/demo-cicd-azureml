"""Entrainement d'un modele de detection de defaut de paiement."""
import argparse

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def separer_x_y(df, cible="default"):
    """Separe les variables explicatives (X) et la cible (y)."""
    X = df.drop(columns=[cible]).astype(float)
    y = df[cible]
    return X, y


def entrainer(X, y, class_weight=None):
    """Cree et entraine le modele."""
    modele = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight=class_weight, max_iter=1000),
    )
    modele.fit(X, y)
    return modele


def evaluer(modele, X, y):
    """Calcule l'accuracy et le recall."""
    predictions = modele.predict(X)
    return {
        "accuracy": accuracy_score(y, predictions),
        "recall": recall_score(y, predictions),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--class_weight", type=str, default="none")
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    class_weight = None if args.class_weight == "none" else "balanced"

    df = pd.read_csv(args.data, index_col=0)
    X, y = separer_x_y(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    modele = entrainer(X_train, y_train, class_weight)
    resultats = evaluer(modele, X_test, y_test)

    mlflow.log_param("class_weight", args.class_weight)
    mlflow.log_metrics(resultats)
    mlflow.sklearn.log_model(modele, "model")
    print(resultats)


if __name__ == "__main__":
    main()
