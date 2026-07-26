# Prédiction du décrochage scolaire

Modèle de classification binaire prédisant le risque de décrochage d'un élève à partir de ses caractéristiques académiques et sociales, déployé en application web Flask.

## Objectif

Identifier en amont les élèves à risque d'échec scolaire, à partir de données académiques (notes, absences, échecs passés) et sociales (contexte familial, habitudes de vie), afin de permettre une intervention préventive.

## Dataset

[UCI Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/student+performance) — 649 élèves (cours de portugais), 33 variables académiques et sociales.

La target `dropout` est créée à partir de la note finale (`G3 < 10` → décrochage).

## Méthodologie

1. **Analyse exploratoire (EDA)** : distribution des variables, valeurs manquantes, croisement avec la target
2. **Feature engineering** : agrégation (`moyenne_G1_G2`, `score_social`, `score_soutien`) et interactions (`studytime × failures`, `goout × Walc`)
3. **Encodage** : binaire, one-hot pour les variables catégorielles
4. **Sélection de variables** : Chi² et Mutual Information
5. **Modélisation** : LogisticRegression, RandomForest, XGBoost
6. **Détection et correction d'overfitting** : comparaison systématique des scores train/test
7. **Optimisation** : GridSearchCV, ajustement du seuil de décision (precision_recall_curve)
8. **Fusion de modèles** : VotingClassifier
9. **Déploiement** : application web Flask

## Résultats

| Modèle | Recall (dropout) | Écart train/test |
|---|---|---|
| LogisticRegression | 58% | sain |
| LogisticRegression + seuil optimisé (0.45) | **77%** | - |

Seuil de décision volontairement abaissé à 0.45 (au lieu de 0.5) : dans ce contexte, un faux positif (fausse alerte) est moins coûteux qu'un faux négatif (élève à risque non détecté).

## Stack technique

Python · Pandas · Scikit-learn · XGBoost · Flask

## Structure du projet

```
├── app.py                    # Backend Flask
├── templates/index.html      # Formulaire web
├── model_dropout.joblib      # Modèle entraîné
├── model_columns.joblib      # Ordre des colonnes attendues
├── requirements.txt
└── *.ipynb                   # Notebooks d'analyse et de modélisation
```

## Installation et lancement

```bash
pip install -r requirements.txt
python app.py
```

Puis ouvrir `http://127.0.0.1:5000` dans le navigateur.

## Limites et pistes d'amélioration

- Dataset de taille modeste (649 lignes)
- Pas d'information sur les notes G1/G2 disponible en amont dans un cas réel d'usage préventif
- Pourrait bénéficier d'une fusion avec le dataset "mathématiques" du même corpus pour augmenter le volume de données
