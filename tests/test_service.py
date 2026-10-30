import numpy as np
import pytest

import cluster_service as svc

HEALTHY = {
    "HighBP": 0, "HighChol": 0, "CholCheck": 1, "BMI": 24, "Smoker": 0, "Stroke": 0,
    "HeartDiseaseorAttack": 0, "PhysActivity": 1, "Fruits": 1, "Veggies": 1,
    "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0, "GenHlth": 2,
    "MentHlth": 0, "PhysHlth": 0, "DiffWalk": 0, "Sex": 0, "Age": 5, "Education": 6, "Income": 8,
}
SICK = {**HEALTHY, "HighBP": 1, "HighChol": 1, "HeartDiseaseorAttack": 1, "DiffWalk": 1,
        "GenHlth": 5, "PhysHlth": 25, "BMI": 38, "Age": 11}


@pytest.fixture(scope="module")
def artifact():
    return svc.load_artifact()


def test_artifact_matches_application(artifact):
    assert tuple(artifact["selected_features"]) == svc.FEATURES
    assert artifact["best_model_name"] == "K-Means"


def test_assignment_matches_kmeans_predict(artifact):
    for profile in (HEALTHY, SICK, {**SICK, "CholCheck": 0}):
        result = svc.predict_cluster(profile, artifact)
        import pandas as pd
        scaled = pd.DataFrame(artifact["scaler"].transform(pd.DataFrame([profile], columns=list(svc.FEATURES))), columns=list(svc.FEATURES))
        assert result.cluster_id == int(artifact["model"].predict(scaled)[0])
        assert result.distance <= result.runner_up_distance


def test_missing_variables_are_imputed_and_reported(artifact):
    partial = {k: v for k, v in HEALTHY.items() if k not in ("BMI", "Income")}
    assert svc.predict_cluster(partial, artifact).imputed == ("BMI", "Income")


def test_blank_strings_count_as_missing(artifact):
    result = svc.predict_cluster({**HEALTHY, "BMI": "", "Age": "  "}, artifact)
    assert result.imputed == ("BMI", "Age")


def test_minimum_number_of_variables(artifact):
    with pytest.raises(ValueError, match="au moins"):
        svc.predict_cluster({"BMI": 27, "Age": 9}, artifact)
    five = dict(list(HEALTHY.items())[:svc.MIN_PROVIDED])
    assert svc.predict_cluster(five, artifact).imputed


@pytest.mark.parametrize("patch, message", [
    ({"Unknown": 1}, "inconnues"),
    ({"HighBP": 2}, "0 ou 1"),
    ({"BMI": 5}, "comprise"),
    ({"MentHlth": 31}, "comprise"),
    ({"Age": 4.5}, "entier"),
    ({"BMI": "abc"}, "numérique"),
    ({"BMI": float("inf")}, "fini"),
])
def test_invalid_values_are_rejected(artifact, patch, message):
    with pytest.raises(ValueError, match=message):
        svc.predict_cluster({**HEALTHY, **patch}, artifact)


def test_decimal_bmi_is_accepted(artifact):
    assert svc.predict_cluster({**HEALTHY, "BMI": 27.5}, artifact)


def test_confidence_levels():
    assert svc.confidence_level(0.05, 0) == "low"
    assert svc.confidence_level(0.15, 0) == "medium"
    assert svc.confidence_level(0.60, 0) == "high"
    assert svc.confidence_level(0.60, svc.MAX_IMPUTED_FOR_HIGH + 1) == "medium"
    assert svc.confidence_level(0.60, svc.MAX_IMPUTED_FOR_MEDIUM + 1) == "low"


def test_cholcheck_alone_moves_a_profile_to_cluster_0(artifact):
    """Comportement connu, documenté dans le README : CholCheck = 0 définit à lui seul le cluster 0."""
    assert svc.predict_cluster(HEALTHY, artifact).cluster_id == 2
    assert svc.predict_cluster({**HEALTHY, "CholCheck": 0}, artifact).cluster_id == 0


def test_sick_profile_without_cholcheck_is_a_near_tie(artifact):
    result = svc.predict_cluster({**SICK, "CholCheck": 0}, artifact)
    assert result.cluster_id == 0 and result.confidence == "low"
