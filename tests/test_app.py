import json

import pytest

import app as application
import cluster_service as svc
from tests.test_service import HEALTHY

client = application.app.test_client()


def post(data):
    return client.post("/", data=data)


def test_pages_and_health():
    for path in ("/", "/groupes", "/methodologie"):
        assert client.get(path).status_code == 200, path
    assert client.get("/health").get_json() == {"status": "ok", "ready": True}


def test_plotly_only_loaded_on_groups_page():
    assert b"plotly" not in client.get("/").data
    assert b"plotly" in client.get("/groupes").data


def test_form_fields_are_optional():
    html = client.get("/").data.decode()
    assert " required" not in html
    assert html.count('<option value="" ') >= 14  # un « Non renseigné » par liste déroulante


def test_complete_form_gives_named_cluster_with_confidence():
    html = post(HEALTHY).data.decode()
    assert "RÉSULTAT DE L’AFFECTATION" in html
    assert "Cluster 2 · " in html
    assert "Deuxième groupe le plus proche" in html
    assert "non renseignée" not in html


def test_missing_fields_are_listed_in_result():
    html = post({k: v for k, v in HEALTHY.items() if k not in ("BMI", "Income")}).data.decode()
    assert "2 variables non renseignées" in html and "IMC" in html


def test_too_few_fields_gives_readable_error():
    html = post({"BMI": 27, "Age": 9}).data.decode()
    assert "au moins 5" in html and "RÉSULTAT DE L’AFFECTATION" not in html


def test_invalid_form_value_is_reported():
    assert "doit être comprise" in post({**HEALTHY, "BMI": 500}).data.decode()


def test_api():
    ok = client.post("/api/predict", json=HEALTHY)
    assert ok.status_code == 200
    body = ok.get_json()
    assert body["cluster_id"] == 2 and body["confidence"] in {"high", "medium", "low"} and body["imputed"] == []
    assert client.post("/api/predict", json={**HEALTHY, "Nope": 1}).status_code == 400
    assert client.post("/api/predict", json={"BMI": 27}).status_code == 400
    assert client.post("/api/predict", data="not json").status_code == 400


def test_groups_payload_covers_all_features():
    heatmap = application.GROUPS["heatmap"]
    assert len(heatmap["y"]) == len(svc.FEATURES) == 21
    assert len(heatmap["x"]) == 3


def test_cluster_names_still_match_the_model_profiles():
    """Si le modèle est ré-entraîné, les noms et notes de app.py doivent être revus."""
    dashboard = json.loads(application.DASHBOARD_PATH.read_text(encoding="utf-8"))
    profiles = {p["cluster_id"]: p["values"] for p in dashboard["cluster_profiles"]}
    assert set(profiles) == set(application.CLUSTER_NAMES)
    assert profiles[0]["CholCheck"] == 0                                     # « Sans contrôle du cholestérol »
    assert all(profiles[i]["CholCheck"] > 0.95 for i in (1, 2))
    health = {i: profiles[i]["GenHlth"] for i in profiles}
    assert health[1] == max(health.values()) and health[2] == min(health.values())  # « dégradée » / « plutôt bonne »
    sizes = {s["cluster_id"]: s["count"] for s in dashboard["cluster_sizes"]}
    assert sizes[2] == max(sizes.values())                                   # « majoritaire »
