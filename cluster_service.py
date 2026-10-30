"""Affectation d'un profil au groupe K-Means sauvegardé.

Le module charge l'artefact produit par le notebook et n'entraîne rien. C'est la
seule source de vérité pour la liste des variables et leurs domaines de valeurs :
l'application web et l'API s'appuient sur les mêmes contrôles.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

ARTIFACT_PATH = Path(os.environ.get("MODEL_ARTIFACT_PATH", Path(__file__).resolve().parent / "model_cdc_diabetes.joblib"))

# Ordre exact des colonnes utilisées à l'entraînement.
FEATURES: tuple[str, ...] = (
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
)
BINARY_FEATURES = frozenset({
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack",
    "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
    "NoDocbcCost", "DiffWalk", "Sex",
})
BOUNDS: dict[str, tuple[float, float]] = {
    "BMI": (12, 98), "GenHlth": (1, 5), "MentHlth": (0, 30), "PhysHlth": (0, 30),
    "Age": (1, 13), "Education": (1, 6), "Income": (1, 8),
}
DECIMAL_FEATURES = frozenset({"BMI"})  # les autres variables bornées sont des codes entiers

# Nombre minimal de variables renseignées : en dessous, l'imputation (valeur la plus
# fréquente à l'entraînement) décide presque seule du résultat.
MIN_PROVIDED = 5

# Écart relatif entre le centroïde le plus proche et le suivant, (d2 - d1) / d1.
# Seuils heuristiques, non calibrés : ils signalent un résultat proche d'une égalité.
MARGIN_LOW, MARGIN_HIGH = 0.10, 0.25
MAX_IMPUTED_FOR_HIGH = 10   # au-delà : confiance plafonnée à « modérée »
MAX_IMPUTED_FOR_MEDIUM = 15  # au-delà : confiance « faible » (moins de 6 variables saisies)


@dataclass(frozen=True)
class Assignment:
    cluster_id: int
    distance: float            # distance au centroïde le plus proche (espace standardisé)
    runner_up_id: int
    runner_up_distance: float
    margin_pct: float          # écart relatif au deuxième centroïde, en %
    confidence: str            # "high" | "medium" | "low"
    imputed: tuple[str, ...]   # variables absentes, remplacées par la valeur d'entraînement


def load_artifact(path: str | Path = ARTIFACT_PATH) -> dict[str, Any]:
    """Charge l'artefact et vérifie qu'il correspond au code de l'application."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Artefact introuvable : {path}")
    artifact = joblib.load(path)
    missing = {"scaler", "model", "best_model_name", "selected_features", "fill_values"} - set(artifact)
    if missing:
        raise ValueError("Artefact incomplet : " + ", ".join(sorted(missing)))
    if artifact["best_model_name"] != "K-Means":
        raise ValueError(f"Seul K-Means est pris en charge (artefact : {artifact['best_model_name']!r}).")
    if tuple(artifact["selected_features"]) != FEATURES:
        raise ValueError("La liste ordonnée des variables de l'artefact ne correspond pas à l'application.")
    if set(artifact["fill_values"]) != set(FEATURES):
        raise ValueError("Valeurs d'imputation incohérentes avec la liste des variables.")
    return artifact


def _clean(feature: str, raw: Any) -> float | None:
    """Convertit une valeur et la valide ; None = valeur absente."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"La variable {feature} doit être numérique.") from None
    if not np.isfinite(value):
        raise ValueError(f"La variable {feature} doit être un nombre fini.")
    if feature in BINARY_FEATURES and value not in (0, 1):
        raise ValueError(f"La variable {feature} doit valoir 0 ou 1.")
    if feature in BOUNDS:
        low, high = BOUNDS[feature]
        if not low <= value <= high:
            raise ValueError(f"La variable {feature} doit être comprise entre {low} et {high}.")
        if feature not in DECIMAL_FEATURES and not value.is_integer():
            raise ValueError(f"La variable {feature} doit être un code entier.")
    return value


def confidence_level(margin: float, n_imputed: int) -> str:
    level = "low" if margin < MARGIN_LOW else "medium" if margin < MARGIN_HIGH else "high"
    if n_imputed > MAX_IMPUTED_FOR_MEDIUM:
        return "low"
    if level == "high" and n_imputed > MAX_IMPUTED_FOR_HIGH:
        return "medium"
    return level


def predict_cluster(values: Mapping[str, Any], artifact: dict[str, Any]) -> Assignment:
    """Affecte un profil au centroïde le plus proche.

    Les variables absentes sont imputées avec les valeurs mémorisées à l'entraînement ;
    les variables inconnues et les valeurs hors domaine sont refusées.
    """
    if not isinstance(values, Mapping):
        raise TypeError("Les valeurs doivent être fournies sous forme de dictionnaire.")
    unknown = sorted(set(values) - set(FEATURES))
    if unknown:
        raise ValueError("Variables inconnues : " + ", ".join(unknown))

    cleaned = {f: _clean(f, values.get(f)) for f in FEATURES}
    provided = [f for f, v in cleaned.items() if v is not None]
    if len(provided) < MIN_PROVIDED:
        raise ValueError(
            f"Renseignez au moins {MIN_PROVIDED} variables pour obtenir une affectation "
            f"({len(provided)} fournie(s))."
        )
    imputed = tuple(f for f in FEATURES if cleaned[f] is None)
    row = {f: (artifact["fill_values"][f] if cleaned[f] is None else cleaned[f]) for f in FEATURES}

    scaled = pd.DataFrame(artifact["scaler"].transform(pd.DataFrame([row], columns=list(FEATURES))), columns=list(FEATURES))
    distances = artifact["model"].transform(scaled)[0]
    order = np.argsort(distances)
    first, second = int(order[0]), int(order[1])
    margin = float((distances[second] - distances[first]) / max(distances[first], 1e-9))
    return Assignment(
        cluster_id=first,
        distance=float(distances[first]),
        runner_up_id=second,
        runner_up_distance=float(distances[second]),
        margin_pct=100 * margin,
        confidence=confidence_level(margin, len(imputed)),
        imputed=imputed,
    )
