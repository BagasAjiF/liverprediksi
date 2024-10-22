# Flask app definition
from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

# Load model
with open("model/best_rf_model.pkl", 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

# Home page route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Handle form inputs and return the prediction result
        age = int(request.form['age'])
        gender = 1 if request.form['gender'] == 'Male' else 0

        total_bilirubin_range = request.form['total_bilirubin']
        direct_bilirubin_range = request.form['direct_bilirubin']
        alkaline_phosphotase_range = request.form['alkaline_phosphotase']
        alamine_aminotransferase_range = request.form['alamine_aminotransferase']
        aspartate_aminotransferase_range = request.form['aspartate_aminotransferase']
        total_proteins_range = request.form['total_proteins']
        albumin_range = request.form['albumin']
        albumin_globulin_ratio_range = request.form['albumin_globulin_ratio']

        # Convert ranges into numerical values for prediction
        total_bilirubin = (0.1 + 1.2) / 2 if total_bilirubin_range == "0.1-1.2" else (1.3 + 2.0) / 2 if total_bilirubin_range == "1.3-2.0" else (2.1 + 5.0) / 2 if total_bilirubin_range == "2.1-5.0" else 6.0
        direct_bilirubin = (0.0 + 0.3) / 2 if direct_bilirubin_range == "0.0-0.3" else (0.4 + 1.0) / 2 if direct_bilirubin_range == "0.4-1.0" else 1.5
        alkaline_phosphotase = (44 + 147) / 2 if alkaline_phosphotase_range == "44-147" else (148 + 300) / 2 if alkaline_phosphotase_range == "148-300" else 350
        alamine_aminotransferase = (7 + 56) / 2 if alamine_aminotransferase_range == "7-56" else (57 + 100) / 2 if alamine_aminotransferase_range == "57-100" else 150
        aspartate_aminotransferase = (10 + 40) / 2 if aspartate_aminotransferase_range == "10-40" else (41 + 100) / 2 if aspartate_aminotransferase_range == "41-100" else 150
        total_proteins = (6.0 + 8.3) / 2 if total_proteins_range == "6.0-8.3" else (8.4 + 10.0) / 2 if total_proteins_range == "8.4-10.0" else 12.0
        albumin = (3.5 + 5.0) / 2 if albumin_range == "3.5-5.0" else (5.1 + 6.0) / 2 if albumin_range == "5.1-6.0" else 7.0
        albumin_globulin_ratio = (1.0 + 2.1) / 2 if albumin_globulin_ratio_range == "1.0-2.1" else 2.5

        # Prepare data for prediction
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Total_Bilirubin': [total_bilirubin],
            'Direct_Bilirubin': [direct_bilirubin],
            'Alkaline_Phosphotase': [alkaline_phosphotase],
            'Alamine_Aminotransferase': [alamine_aminotransferase],
            'Aspartate_Aminotransferase': [aspartate_aminotransferase],
            'Total_Protiens': [total_proteins],
            'Albumin': [albumin],
            'Albumin_and_Globulin_Ratio': [albumin_globulin_ratio]
        })

        # Make prediction
        prediction_proba = model.predict_proba(input_data)
        result = "Pasien Terkena Penyakit Liver" if prediction_proba[0][1] >= 0.7 else "Pasien Tidak Terkena Penyakit Liver"

        return jsonify(prediction=result)

    return render_template('predict.html')  # Form for prediction

# Other routes...
@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/konsultasi')
def konsultasi():
    return render_template('konsultasi.html')

@app.route('/gambar')
def gambar():
    return render_template('gambar.html')

if __name__ == '__main__':
    app.run(debug=True)
