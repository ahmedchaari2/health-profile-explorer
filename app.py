"""CDC Health Profile Explorer — application Flask de clustering non supervisé.

Charge le modèle K-Means sauvegardé par le notebook ; ne réentraîne rien et ne remplace
pas une évaluation clinique. Aucune valeur saisie n'est enregistrée.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from flask import Flask, render_template, request

import cluster_service as svc

BASE_DIR = Path(__file__).resolve().parent
DASHBOARD_PATH = Path(os.environ.get("DASHBOARD_DATA_PATH", BASE_DIR / "cluster_dashboard_data.json"))

app = Flask(__name__)

# --------------------------------------------------------------------------- libellés

# Noms descriptifs des groupes de l'artefact actuel. tests/test_app.py vérifie que les
# profils correspondent toujours à ces noms : à revoir si le modèle est ré-entraîné.
CLUSTER_NAMES = {
    0: "Sans contrôle du cholestérol",
    1: "Santé dégradée, facteurs de risque cumulés",
    2: "Santé plutôt bonne (groupe majoritaire)",
}
CLUSTER_NOTES = {
    0: "Ce groupe est défini presque uniquement par une réponse : aucune personne n’y déclare de contrôle du cholestérol "
       "(environ 4 % de l’échantillon). K-Means isole cette variable rare plutôt qu’un profil de santé : les autres réponses "
       "pèsent peu dans l’affectation, et son faible taux de diabète ne rassure pas sur une personne donnée.",
}
CONFIDENCE_LABELS = {"high": "élevée", "medium": "modérée", "low": "faible"}
SMALL_GROUP = 200  # en dessous, les pourcentages d'un groupe reposent sur peu de personnes

FEATURE_META = {
    "HighBP": ("Hypertension", "binary"), "HighChol": ("Cholestérol élevé", "binary"),
    "CholCheck": ("Contrôle du cholestérol", "binary"), "BMI": ("IMC", "numeric"),
    "Smoker": ("Tabagisme", "binary"), "Stroke": ("Antécédent d’AVC", "binary"),
    "HeartDiseaseorAttack": ("Maladie cardiaque ou infarctus", "binary"),
    "PhysActivity": ("Activité physique", "binary"), "Fruits": ("Consommation de fruits", "binary"),
    "Veggies": ("Consommation de légumes", "binary"), "HvyAlcoholConsump": ("Consommation élevée d’alcool", "binary"),
    "AnyHealthcare": ("Accès aux soins", "binary"), "NoDocbcCost": ("Renoncement aux soins pour coût", "binary"),
    "GenHlth": ("Santé générale (code)", "ordinal"), "MentHlth": ("Jours de mauvaise santé mentale", "numeric"),
    "PhysHlth": ("Jours de mauvaise santé physique", "numeric"), "DiffWalk": ("Difficulté à marcher", "binary"),
    "Sex": ("Sexe (code)", "ordinal"), "Age": ("Classe d’âge (code)", "ordinal"),
    "Education": ("Éducation (code)", "ordinal"), "Income": ("Revenu (code)", "ordinal"),
}


def cluster_name(cluster_id: int) -> str:
    return CLUSTER_NAMES.get(cluster_id, "")


def cluster_title(cluster_id: int) -> str:
    name = cluster_name(cluster_id)
    return f"Cluster {cluster_id} · {name}" if name else f"Cluster {cluster_id}"


def fmt(key: str, value: float) -> str:
    """Valeur moyenne d'un groupe, formatée selon le type de variable."""
    if FEATURE_META[key][1] == "binary":
        return f"{100 * value:.1f} %"
    if key == "GenHlth":
        return f"{value:.2f} / 5"
    if key in ("MentHlth", "PhysHlth"):
        return f"{value:.1f} j"
    return f"{value:.1f}" if key == "BMI" else f"{value:.2f}"


# --------------------------------------------------------------------------- formulaire

YES_NO = [{"value": "0", "label": "Non"}, {"value": "1", "label": "Oui"}]


def _codes(pairs):
    return [{"value": str(v), "label": label} for v, label in pairs]


def _yes_no(key, label, help_text):
    return {"key": key, "label": label, "kind": "select", "options": YES_NO, "help": help_text}


def _select(key, label, pairs, help_text):
    return {"key": key, "label": label, "kind": "select", "options": _codes(pairs), "help": help_text}


def _number(key, label, step, placeholder, help_text):
    low, high = svc.BOUNDS[key]
    return {"key": key, "label": label, "kind": "number", "min": low, "max": high, "step": step,
            "placeholder": placeholder, "help": help_text}


FORM_GROUPS = [
    {"title": "Santé et antécédents", "description": "Indicateurs de santé déclarés dans le questionnaire.", "fields": [
        _yes_no("HighBP", "Hypertension", "1 = hypertension déclarée."),
        _yes_no("HighChol", "Cholestérol élevé", "1 = cholestérol élevé déclaré."),
        _yes_no("CholCheck", "Contrôle du cholestérol", "1 = contrôle effectué. Cette réponse pèse fortement sur le groupe obtenu."),
        _yes_no("Stroke", "Antécédent d’AVC", "1 = AVC déclaré."),
        _yes_no("HeartDiseaseorAttack", "Maladie cardiaque ou infarctus", "1 = maladie cardiaque ou infarctus déclaré."),
        _select("GenHlth", "Santé générale", [(1, "1 — Excellente"), (2, "2 — Très bonne"), (3, "3 — Bonne"), (4, "4 — Passable"), (5, "5 — Mauvaise")], "Code ordinal : 1 = excellente, 5 = mauvaise."),
        _yes_no("DiffWalk", "Difficulté à marcher ou monter des escaliers", "1 = difficulté déclarée."),
    ]},
    {"title": "Habitudes de vie", "description": "Caractéristiques de mode de vie utilisées par le modèle.", "fields": [
        _number("BMI", "IMC (BMI)", "0.1", "Ex. 27", "Indice de masse corporelle, entre 12 et 98."),
        _yes_no("Smoker", "Tabagisme", "1 = a fumé au moins 100 cigarettes (codage du jeu)."),
        _yes_no("PhysActivity", "Activité physique", "1 = activité physique déclarée."),
        _yes_no("Fruits", "Consommation de fruits", "1 = au moins un fruit par jour."),
        _yes_no("Veggies", "Consommation de légumes", "1 = au moins un légume par jour."),
        _yes_no("HvyAlcoholConsump", "Consommation élevée d’alcool", "1 = consommation élevée déclarée."),
    ]},
    {"title": "Accès aux soins et bien-être", "description": "Recours aux soins et santé physique ou mentale.", "fields": [
        _yes_no("AnyHealthcare", "Couverture ou accès aux soins", "1 = couverture de santé déclarée."),
        _yes_no("NoDocbcCost", "Renoncement aux soins pour raisons financières", "1 = consultation impossible à cause du coût."),
        _number("MentHlth", "Jours de mauvaise santé mentale", "1", "0 à 30", "Nombre de jours sur les 30 derniers jours."),
        _number("PhysHlth", "Jours de mauvaise santé physique", "1", "0 à 30", "Nombre de jours sur les 30 derniers jours."),
    ]},
    {"title": "Caractéristiques sociodémographiques", "description": "Codes catégoriels tels qu’ils sont utilisés à l’entraînement.", "fields": [
        _select("Sex", "Sexe (code du jeu)", [(0, "0 — Femme"), (1, "1 — Homme")], "Valeur encodée du jeu de données, pas une appréciation de genre."),
        _select("Age", "Classe d’âge", [(1, "18–24 ans"), (2, "25–29 ans"), (3, "30–34 ans"), (4, "35–39 ans"), (5, "40–44 ans"), (6, "45–49 ans"), (7, "50–54 ans"), (8, "55–59 ans"), (9, "60–64 ans"), (10, "65–69 ans"), (11, "70–74 ans"), (12, "75–79 ans"), (13, "80 ans et plus")], "Code ordinal Age : 1 à 13."),
        _select("Education", "Niveau d’éducation (code)", [(1, "1 — Aucun / maternelle"), (2, "2 — Primaire"), (3, "3 — Secondaire incomplet"), (4, "4 — Diplôme secondaire ou équivalent"), (5, "5 — Études supérieures, 1 à 3 ans"), (6, "6 — Études supérieures, 4 ans ou plus")], "Code ordinal du jeu de données."),
        _select("Income", "Catégorie de revenu (code)", [(1, "1 — Moins de 10 000 $"), (2, "2 — 10 000 à moins de 15 000 $"), (3, "3 — 15 000 à moins de 20 000 $"), (4, "4 — 20 000 à moins de 25 000 $"), (5, "5 — 25 000 à moins de 35 000 $"), (6, "6 — 35 000 à moins de 50 000 $"), (7, "7 — 50 000 à moins de 75 000 $"), (8, "8 — 75 000 $ ou plus")], "Code ordinal, pas un montant exact."),
    ]},
]

# --------------------------------------------------------------------------- chargement

ARTIFACT: dict[str, Any] | None = None
DASHBOARD: dict[str, Any] | None = None
GROUPS: dict[str, Any] | None = None  # données minimales envoyées au navigateur
READY_ERROR: str | None = None


def _sorted_heatmap(dashboard: dict[str, Any]) -> dict[str, Any]:
    """Écart standardisé de chaque groupe à la moyenne, pour les 21 variables, triées par écart maximal."""
    profiles, means, stds = dashboard["cluster_profiles"], dashboard["overall_means"], dashboard["overall_stds"]
    rows = []
    for key in svc.FEATURES:
        std = stds.get(key) or 0
        z = [(p["values"][key] - means[key]) / std if std else 0.0 for p in profiles]
        rows.append((max(abs(v) for v in z), key, z, [fmt(key, p["values"][key]) for p in profiles]))
    rows.sort(key=lambda r: r[0], reverse=True)
    return {
        "x": [cluster_title(p["cluster_id"]) for p in profiles],
        "y": [FEATURE_META[r[1]][0] for r in rows],
        "z": [[round(v, 3) for v in r[2]] for r in rows],
        "text": [r[3] for r in rows],
    }


def _load() -> None:
    global ARTIFACT, DASHBOARD, GROUPS, READY_ERROR
    errors = []
    try:
        ARTIFACT = svc.load_artifact()
    except Exception as exc:  # message explicite sur la page plutôt qu'une erreur 500
        errors.append(f"Modèle : {exc}")
    try:
        DASHBOARD = json.loads(DASHBOARD_PATH.read_text(encoding="utf-8"))
        GROUPS = {
            "sizes": [{**s, "title": cluster_title(s["cluster_id"])} for s in DASHBOARD["cluster_sizes"]],
            "heatmap": _sorted_heatmap(DASHBOARD),
        }
    except Exception as exc:
        DASHBOARD = GROUPS = None
        errors.append(f"Statistiques : {type(exc).__name__}: {exc}")
    READY_ERROR = " · ".join(errors) or None


_load()


METRIC_DEFAULTS = {"n_clusters": "—", "silhouette": None, "calinski_harabasz": None, "davies_bouldin": None, "largest_cluster_pct": None}


def is_ready() -> bool:
    return ARTIFACT is not None and DASHBOARD is not None and READY_ERROR is None


def _by_id(items_key: str, cluster_id: int) -> dict[str, Any]:
    return next((i for i in (DASHBOARD or {}).get(items_key, []) if int(i["cluster_id"]) == cluster_id), {})


# --------------------------------------------------------------------------- lecture d'un groupe

HIGHLIGHTS = [
    ("BMI", "IMC moyen", "Moyenne de l’IMC dans le groupe"),
    ("HighBP", "Hypertension déclarée", "Proportion observée dans le groupe"),
    ("HighChol", "Cholestérol élevé déclaré", "Proportion observée dans le groupe"),
    ("PhysActivity", "Activité physique déclarée", "Proportion observée dans le groupe"),
    ("GenHlth", "Santé générale", "Moyenne du code (1 = excellente, 5 = mauvaise)"),
]


def highlights(profile: dict[str, Any]) -> list[dict[str, str]]:
    values, overall = profile.get("values", {}), DASHBOARD["overall_means"]
    return [{"label": label, "value": fmt(key, values[key]), "overall": fmt(key, overall[key]), "note": note}
            for key, label, note in HIGHLIGHTS if values.get(key) is not None]


def describe(profile: dict[str, Any]) -> str:
    """Les trois écarts les plus marqués du groupe par rapport à l'échantillon."""
    values, overall, stds = profile.get("values", {}), DASHBOARD["overall_means"], DASHBOARD["overall_stds"]
    gaps = []
    for key in svc.FEATURES:
        if values.get(key) is None or not stds.get(key):
            continue
        z = (values[key] - overall[key]) / stds[key]
        if abs(z) >= 0.35:
            gaps.append((abs(z), key))
    if not gaps:
        return "Ce groupe ne présente pas d’écart marqué sur les indicateurs résumés ; consultez les profils comparés."
    top = sorted(gaps, reverse=True)[:3]
    parts = [f"{FEATURE_META[k][0]} : {fmt(k, values[k])} (échantillon : {fmt(k, overall[k])})" for _, k in top]
    return "Écarts les plus marqués avec l’échantillon — " + " ; ".join(parts) + "."


def build_result(assignment: svc.Assignment) -> dict[str, Any]:
    cid, rid = assignment.cluster_id, assignment.runner_up_id
    profile, size = _by_id("cluster_profiles", cid), _by_id("cluster_sizes", cid)
    count = size.get("count")
    return {
        **asdict(assignment),
        "title": cluster_title(cid),
        "note": CLUSTER_NOTES.get(cid),
        "confidence_label": CONFIDENCE_LABELS[assignment.confidence],
        "runner_up_title": cluster_title(rid),
        "description": describe(profile) if profile else None,
        "highlights": highlights(profile) if profile else [],
        "count": count,
        "percentage": size.get("percentage"),
        "small_group": count is not None and count < SMALL_GROUP,
        "diabetes_pct": profile.get("diabetes_prevalence_pct"),
        "imputed_labels": [FEATURE_META[k][0] for k in assignment.imputed],
    }


# --------------------------------------------------------------------------- pages

def render(page: str, **overrides):
    context = {
        "page": page, "ready": is_ready(), "ready_error": READY_ERROR, "model_name": "K-Means",
        "metrics": {**METRIC_DEFAULTS, **(DASHBOARD or {}).get("metrics", {})},
        "sample_size": (DASHBOARD or {}).get("sample_size", "—"),
        "versions": (DASHBOARD or {}).get("runtime_versions", {}),
        "feature_count": len(svc.FEATURES), "min_provided": svc.MIN_PROVIDED,
        "cluster_notes": [{"title": cluster_title(i), "text": t} for i, t in CLUSTER_NOTES.items()],
        "groups": GROUPS if page == "groups" else None,
        "form_groups": FORM_GROUPS, "form_values": {}, "prediction": None, "error_message": None,
    }
    return render_template("index.html", **{**context, **overrides})


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render("form")
    form_values = request.form.to_dict()
    if not is_ready():
        return render("form", form_values=form_values,
                      error_message="Le modèle ou les statistiques ne sont pas disponibles. Vérifiez les artefacts de déploiement.")
    try:
        assignment = svc.predict_cluster({k: v for k, v in form_values.items() if v.strip()}, ARTIFACT)
    except ValueError as exc:
        return render("form", form_values=form_values, error_message=str(exc))
    return render("form", form_values=form_values, prediction=build_result(assignment))


@app.route("/groupes")
def groups():
    return render("groups")


@app.route("/methodologie")
def methodology():
    return render("method")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Corps JSON : dictionnaire de variables (au moins 5). Renvoie le groupe et son indice de proximité."""
    if not is_ready():
        return {"error": "Application non prête", "details": READY_ERROR}, 503
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return {"error": "Le corps doit être un objet JSON de type dictionnaire."}, 400
    try:
        result = build_result(svc.predict_cluster(payload, ARTIFACT))
    except (ValueError, TypeError) as exc:
        return {"error": str(exc)}, 400
    keep = ("cluster_id", "title", "distance", "runner_up_id", "runner_up_distance", "margin_pct",
            "confidence", "imputed", "count", "percentage", "diabetes_pct", "description")
    return {**{k: result[k] for k in keep}, "disclaimer": "Affectation descriptive : ni diagnostic, ni probabilité."}


@app.route("/health")
def health():
    return {"status": "ok" if is_ready() else "degraded", "ready": is_ready()}


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")))
