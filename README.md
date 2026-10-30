# CDC Health Profile Explorer

Application Flask (en français) fondée sur le notebook *CDC Diabetes Health Indicators* (UCI ID 891). Elle charge **un seul modèle, K-Means**, retenu par la règle de sélection du notebook, et ne réentraîne rien. Les valeurs saisies ne sont pas enregistrées.

## Contenu

| Fichier | Rôle |
|---|---|
| `CDC_Diabetes_Clustering_AppReady.ipynb` | Notebook source : exploration, K-Means vs DBSCAN, évaluation, export des deux artefacts |
| `model_cdc_diabetes.joblib` | K-Means ajusté, `StandardScaler`, 21 variables ordonnées, valeurs d’imputation |
| `cluster_dashboard_data.json` | Métriques, effectifs et profils moyens calculés par le notebook |
| `cluster_service.py` | Chargement de l’artefact, validation des entrées, affectation, indice de proximité |
| `app.py`, `templates/`, `static/` | Pages `/` (explorer un profil), `/groupes`, `/methodologie` et API `/api/predict` |
| `tests/` | Tests pytest du service et de l’application |

## Lancer en local

```bash
python -m venv .venv && source .venv/bin/activate     # Windows : .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py                                         # http://127.0.0.1:5000
```

`/health` doit renvoyer `{"ready": true, "status": "ok"}`. Le modèle a été sérialisé avec Python 3.13.5, scikit-learn 1.8.0, NumPy 2.3.5, pandas 2.2.3 et joblib 1.5.3 ; garder ces versions évite les problèmes de compatibilité.

## Tests

```bash
pip install pytest
pytest
```

## Utilisation

Au moins **5 variables** doivent être renseignées (formulaire comme API). Les autres sont remplacées par la valeur la plus fréquente à l’entraînement ; le résultat liste les variables imputées. Les noms de variables inconnus et les valeurs hors domaine sont refusés.

```bash
curl -X POST localhost:5000/api/predict -H "Content-Type: application/json" \
  -d '{"HighBP":0,"HighChol":0,"CholCheck":1,"BMI":27,"GenHlth":2,"Age":9}'
```

La réponse contient le groupe, la distance au centroïde, le deuxième groupe le plus proche, l’écart relatif entre les deux (`margin_pct`) et un niveau de proximité (`high`, `medium`, `low`). Ces seuils sont heuristiques : ce n’est **pas** une probabilité.

## Déployer sur Render

Dépôt GitHub avec le contenu de ce dossier à la racine, puis *New → Web Service*. `render.yaml` fournit la configuration (build `pip install -r requirements.txt`, démarrage `gunicorn app:app --workers 1 --threads 4 --timeout 120`, contrôle de santé `/health`).

## Limites connues

- **Le cluster 0 est défini par une seule variable.** Toutes les personnes qui y figurent déclarent ne pas avoir fait contrôler leur cholestérol (`CholCheck = 0`, environ 4 % de l’échantillon). Changer cette seule réponse fait passer un profil du cluster 2 au cluster 0. Pour corriger cela il faut ré-entraîner sans `CholCheck` ou avec un algorithme adapté aux variables binaires (Gower, K-Prototypes).
- **Structure faible.** Silhouette 0,218 : les groupes se chevauchent beaucoup. Le résultat est exploratoire.
- **Imputation.** Avec peu de variables saisies, le résultat tend vers le groupe majoritaire.
- Clustering sur 3 000 personnes tirées de 253 680 ; distance euclidienne sur variables binaires et ordinales ; données anciennes ; aucune causalité. Le sexe, le revenu et l’éducation sont des variables d’entrée.

## Ré-entraîner le modèle

Réexécuter le notebook, remplacer `model_cdc_diabetes.joblib` et `cluster_dashboard_data.json`, puis revoir `CLUSTER_NAMES` et `CLUSTER_NOTES` dans `app.py`. `tests/test_app.py` échoue si les noms ne correspondent plus aux profils. Le fichier JSON contient aussi `scatter_points`, produit par le notebook et ignoré par l’application.
