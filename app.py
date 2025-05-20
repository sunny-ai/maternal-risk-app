from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load artifacts
model = load_model('pregnancy_risk_model.h5')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Validate and parse blood pressure
            bp = request.form['blood_pressure']
            if '/' not in bp:
                raise ValueError("Use format 120/80 for blood pressure")
                
            systolic, diastolic = map(int, bp.split('/'))
            if systolic <= 50 or diastolic <= 30:  # Basic validation
                raise ValueError("Invalid blood pressure values")

            # Get other form data
            features = [
                float(request.form['age']),
                float(request.form['bmi']),
                float(request.form['gestational_age']),
                int(request.form.get('previous_c_section', 0)),
                int(request.form.get('previous_miscarriages', 0)),
                int(request.form.get('previous_preterm_birth', 0)),
                int(request.form.get('chronic_hypertension', 0)),
                int(request.form.get('diabetes', 0)),
                int(request.form.get('gestational_diabetes', 0)),
                int(request.form.get('preeclampsia_history', 0)),
                int(request.form.get('multiple_pregnancy', 0)),
                int(request.form.get('smoking', 0)),
                int(request.form.get('alcohol_use', 0)),
                int(request.form.get('family_history', 0)),
                float(request.form['hb_level']),
                float(request.form['urine_protein']),
                float(request.form['blood_glucose']),
                systolic,
                diastolic
            ]

            # Preprocess and predict
            scaled_features = scaler.transform([features])
            prediction = model.predict(scaled_features)
            risk_level = encoder.inverse_transform([np.argmax(prediction)])[0]
            confidence = round(np.max(prediction) * 100, 2)

            return render_template('index.html', 
                                prediction=risk_level,
                                probability=confidence)

        except Exception as e:
            return render_template('index.html', error=f"Error: {str(e)}")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)