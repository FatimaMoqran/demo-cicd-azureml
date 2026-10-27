"""Tests automatiques de train.py (donnees inventees, sans Azure)."""
import pandas as pd

from train import entrainer, evaluer, separer_x_y


def creer_donnees():
    """Petit tableau de 8 clients invente pour les tests."""
    return pd.DataFrame({
        "x1": [1, 2, 3, 4, 5, 6, 7, 8],
        "x2": [0, 1, 0, 1, 0, 1, 0, 1],
        "default": [0, 0, 0, 0, 1, 1, 1, 1],
    })


def test_la_cible_est_retiree_de_x():
    X, y = separer_x_y(creer_donnees())
    assert "default" not in X.columns
    assert len(X) == len(y) == 8


def test_le_modele_predit_0_ou_1():
    X, y = separer_x_y(creer_donnees())
    modele = entrainer(X, y)
    predictions = modele.predict(X)
    assert set(predictions) <= {0, 1}


def test_le_recall_est_entre_0_et_1():
    X, y = separer_x_y(creer_donnees())
    modele = entrainer(X, y, class_weight="balanced")
    resultats = evaluer(modele, X, y)
    assert 0 <= resultats["recall"] <= 1
