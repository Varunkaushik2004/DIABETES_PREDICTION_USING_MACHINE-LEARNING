from flask import Flask, request, send_from_directory, make_response
import pickle
import numpy as np

# Tell Flask to look for HTML in current folder
app = Flask(__name__, template_folder='.')

# Load models
logistic_model = pickle.load(open('logistic_model.pkl', 'rb'))
random_forest_model = pickle.load(open('random_forest_model.pkl', 'rb'))

# Serve index.html from current directory
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# Serve CSS from current directory
@app.route('/style.css')
def css():
    return send_from_directory('.', 'style.css')

# Handle form and make prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
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

        model_type = request.form['model']
        if model_type == 'logistic':
            prediction = logistic_model.predict([data])
        else:
            prediction = random_forest_model.predict([data])

        result = "Diabetes Detected" if prediction[0] == 1 else "No Diabetes"

        # Read the index.html and inject result manually
        with open('index.html', 'r', encoding='utf-8') as file:
            html = file.read().replace('{{ result }}', result)

        return make_response(html)

    except Exception as e:
        with open('index.html', 'r', encoding='utf-8') as file:
            html = file.read().replace('{{ result }}', f"Error: {str(e)}")

        return make_response(html)

if __name__ == '__main__':
    app.run(debug=True)
