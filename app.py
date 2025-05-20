from flask import Flask, request, render_template, flash
from flask_sqlalchemy import SQLAlchemy
import joblib
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///instance/maternal.db'
app.config['SECRET_KEY'] = 'your-secret-key'
db = SQLAlchemy(app)

# Configure logging
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=3)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)

# Database Model
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    input_data = db.Column(db.JSON, nullable=False)
    result = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Load ML artifacts
try:
    model = joblib.load('maternal_risk_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('label_encoder.pkl')
except Exception as e:
    app.logger.error(f"Failed to load ML artifacts: {str(e)}")
    raise

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            data = request.form.to_dict()
            
            # Validate required fields
            required_fields = [
                'age', 'bmi', 'blood_pressure', 'gestational_age',
                'previous_c_section', 'previous_miscarriages',
                'previous_preterm_birth', 'chronic_hypertension',
                'diabetes', 'gestational_diabetes', 'preeclampsia_history',
                'multiple_pregnancy', 'smoking', 'alcohol_use',
                'family_history', 'hb_level', 'urine_protein', 'blood_glucose'
            ]
            
            for field in required_fields:
                if not data.get(field):
                    raise ValueError(f"{field.replace('_', ' ').title()} is required")
            
            # Validate blood pressure format
            if '/' not in data['blood_pressure']:
                raise ValueError("Invalid blood pressure format (use 120/80)")
            
            # Process features
            features = [
                float(data['age']),
                float(data['bmi']),
                float(data['gestational_age']),
                int(data['previous_c_section']),
                int(data['previous_miscarriages']),
                int(data['previous_preterm_birth']),
                int(data['chronic_hypertension']),
                int(data['diabetes']),
                int(data['gestational_diabetes']),
                int(data['preeclampsia_history']),
                int(data['multiple_pregnancy']),
                int(data['smoking']),
                int(data['alcohol_use']),
                int(data['family_history']),
                float(data['hb_level']),
                float(data['urine_protein']),
                float(data['blood_glucose']),
                *map(int, data['blood_pressure'].split('/')),
                float(data['bmi']) * float(data['gestational_age']),
                float(data['hb_level']) / float(data['blood_glucose'])
            ]
            
            # Scale features and predict
            sample = np.array(features).reshape(1, -1)
            scaled_sample = scaler.transform(sample)
            prediction = model.predict(scaled_sample)
            risk_level = le.inverse_transform(prediction)[0]
            
            # Log prediction
            new_prediction = Prediction(
                input_data=data,
                result=risk_level
            )
            db.session.add(new_prediction)
            db.session.commit()
            
            return render_template('result.html', prediction=risk_level)
        
        except ValueError as ve:
            flash(str(ve), 'error')
            return render_template('index.html')
        
        except Exception as e:
            app.logger.error(f"Prediction error: {str(e)}")
            db.session.rollback()
            flash('An error occurred during prediction', 'error')
            return render_template('index.html')
    
    return render_template('index.html'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5000)