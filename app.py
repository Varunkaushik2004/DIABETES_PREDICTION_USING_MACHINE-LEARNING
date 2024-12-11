from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load models
logistic_model = pickle.load(open('logistic_model.pkl', 'rb'))
random_forest_model = pickle.load(open('random_forest_model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')  # Serve the HTML dashboard

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    data = [
        float(request.form['Pregnancies']),
        float(request.form['Glucose']),
        float(request.form['BloodPressure']),
        float(request.form['SkinThickness']),
        float(request.form['Insulin']),
        float(request.form['BMI']),
        float(request.form['DiabetesPedigreeFunction']),
        float(request.form['Age'])
    ]

    # Predict using the selected model
    model_type = request.form['model']
    if model_type == 'logistic':
        prediction = logistic_model.predict([data])
    else:  # Random Forest
        prediction = random_forest_model.predict([data])

    # Return result
    result = "Diabetes Detected" if prediction[0] == 1 else "No Diabetes"
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
