from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

application = Flask(__name__)
app = application

# Load model and scaler
ridge_model = pickle.load(open(r"C:\Users\anand\ML FULL\models\ridge.pkl", 'rb'))
st = pickle.load(open(r"C:\Users\anand\ML FULL\models\scaler.pkl", 'rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predictdata", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "POST":
        try:
            # Collect form data
            Temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))
            classes = float(request.form.get('Classes'))
            Region = float(request.form.get('Region'))

            # Create input array
            input_data = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, classes, Region]])

            # Transform input with scaler
            new_data = st.transform(input_data)

            # Predict
            result = ridge_model.predict(new_data)

            return render_template('home.html', result=result[0])
        
        except Exception as e:
            return f"Error occurred: {str(e)}"
    
    return render_template('home.html', result=None)

if __name__ == '__main__':
    app.run(host="0.0.0.0")
