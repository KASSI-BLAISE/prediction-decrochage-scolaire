# App de prédiction du décrochage scolaire

## Installation
```
pip install -r requirements.txt
```

## Lancement
```
python app.py
```
Puis ouvrir http://127.0.0.1:5000 dans le navigateur.

## Fichiers
- app.py : backend Flask (formulaire + prédiction)
- templates/index.html : formulaire web
- model_dropout.joblib : modèle entraîné (LogisticRegression, GridSearchCV)
- model_columns.joblib : liste des colonnes attendues par le modèle, dans l'ordre
