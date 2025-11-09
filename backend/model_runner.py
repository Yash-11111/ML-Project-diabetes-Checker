# model_runner.py
import sys
import pickle
import numpy as np

# Load model
with open("diabetes_model.pkl", "rb") as file:
    model = pickle.load(file)

# Get input features from command line args
# Example order: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
data = np.array([float(x) for x in sys.argv[1:]]).reshape(1, -1)

# Predict
prediction = model.predict(data)[0]

# Output
if prediction == 1:
    print("Diabetic - High Risk")
else:
    print("Non-Diabetic")
