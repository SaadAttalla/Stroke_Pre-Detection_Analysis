from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load your trained model
model = joblib.load('stroke_model.pkl')

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        gender = float(request.form['gender'])
        age = float(request.form['age'])
        hypertension = float(request.form['hypertension'])
        heart_disease = float(request.form['heart_disease'])
        ever_married = float(request.form['ever_married'])
        work_type = float(request.form['work_type'])
        residence_type = float(request.form['Residence_type'])
        avg_glucose_level = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])
        smoking_status = float(request.form['smoking_status'])

        # Make prediction
        features = np.array([[gender, age, hypertension, heart_disease, ever_married,
                              work_type, residence_type, avg_glucose_level, bmi, smoking_status]])
        prediction = model.predict(features)[0]

        result = "Stroke Risk Detected ⚠️" if prediction == 1 else "No Stroke Risk ✅"

        return render_template('index.html', prediction=result)

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=False )


