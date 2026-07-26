import joblib
import pandas as pd
from flask import Flask, render_template, request

model = joblib.load('model_dropout.joblib')
model_columns = joblib.load('model_columns.joblib')

THRESHOLD = 0.45  # seuil optimisé trouvé avec precision_recall_curve

app = Flask(__name__)


def encode_form_data(form):
    """
    Convertit les données texte du formulaire en un DataFrame numérique,
    dans le même ordre de colonnes que celui utilisé à l'entraînement.
    """
    # --- Variables binaires et numériques directes ---
    data = {
        'school': int(form['school']),
        'sex': int(form['sex']),
        'age': int(form['age']),
        'address': int(form['address']),
        'famsize': int(form['famsize']),
        'Pstatus': int(form['Pstatus']),
        'Medu': int(form['Medu']),
        'Fedu': int(form['Fedu']),
        'traveltime': int(form['traveltime']),
        'studytime': int(form['studytime']),
        'failures': int(form['failures']),
        'schoolsup': int(form['schoolsup']),
        'famsup': int(form['famsup']),
        'paid': int(form['paid']),
        'activities': int(form['activities']),
        'nursery': int(form['nursery']),
        'higher': int(form['higher']),
        'internet': int(form['internet']),
        'romantic': int(form['romantic']),
        'famrel': int(form['famrel']),
        'freetime': int(form['freetime']),
        'goout': int(form['goout']),
        'Dalc': int(form['Dalc']),
        'Walc': int(form['Walc']),
        'health': int(form['health']),
        'absences': int(form['absences']),
    }

    # --- Variables d'agrégation / interaction (feature engineering) ---
    # Recréées à partir des valeurs brutes, comme dans le notebook d'entraînement.
    # G1/G2 ne sont pas demandés à l'utilisateur (un futur élève n'a pas encore ces notes),
    # donc on utilise une valeur par défaut neutre (ex: 10) pour moyenne_G1_G2.
    g1 = int(form.get('G1', 10))
    g2 = int(form.get('G2', 10))
    data['moyenne_G1_G2'] = (g1 + g2) / 2
    data['score_social'] = data['freetime'] + data['goout'] + data['Dalc'] + data['Walc']
    data['score_soutien'] = data['schoolsup'] + data['famsup'] + data['paid']
    data['interaction_etude_echec'] = data['studytime'] * data['failures']
    data['interaction_sortie_alcool'] = data['goout'] * data['Walc']
    data['interaction_absence_soutien'] = data['absences'] * (1 - data['schoolsup'])

    # --- One-hot encoding manuel (Mjob, Fjob, reason, guardian) ---
    mjob = form['Mjob']
    for cat in ['health', 'other', 'services', 'teacher']:
        data[f'Mjob_{cat}'] = 1 if mjob == cat else 0

    fjob = form['Fjob']
    for cat in ['health', 'other', 'services', 'teacher']:
        data[f'Fjob_{cat}'] = 1 if fjob == cat else 0

    reason = form['reason']
    for cat in ['home', 'other', 'reputation']:
        data[f'reason_{cat}'] = 1 if reason == cat else 0

    guardian = form['guardian']
    for cat in ['mother', 'other']:
        data[f'guardian_{cat}'] = 1 if guardian == cat else 0

    # --- Construction du DataFrame dans l'ORDRE EXACT attendu par le modèle ---
    df = pd.DataFrame([data])
    df = df[model_columns]
    return df


@app.route('/')
def home():
    return render_template('index.html', result=None)


@app.route('/predict', methods=['POST'])
def predict():
    X = encode_form_data(request.form)

    proba = model.predict_proba(X)[0][1]
    prediction = 1 if proba >= THRESHOLD else 0

    result = {
        'prediction': prediction,
        'proba': round(proba * 100, 1),
        'risque': 'ÉLEVÉ' if prediction == 1 else 'FAIBLE'
    }

    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
