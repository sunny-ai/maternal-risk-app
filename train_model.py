import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib

# Load dataset
df = pd.read_csv('maternal_data.csv')

# Preprocessing
# Split blood pressure into systolic/diastolic
df[['systolic', 'diastolic']] = df['blood_pressure'].str.split('/', expand=True).astype(int)
df.drop('blood_pressure', axis=1, inplace=True)

# Encode target
encoder = LabelEncoder()
df['risk_level'] = encoder.fit_transform(df['risk_level'])

# Split data
X = df.drop('risk_level', axis=1)
y = df['risk_level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(len(encoder.classes_), activation='softmax')
])

model.compile(optimizer=Adam(0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2)

# Save artifacts
model.save('pregnancy_risk_model.h5')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(encoder, 'encoder.pkl')