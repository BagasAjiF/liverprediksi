# app.py
import os
import numpy as np
import pandas as pd
import pickle
import base64
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from werkzeug.utils import secure_filename
from PIL import Image

# Initialize Flask App
app = Flask(__name__)

# Load Liver Disease Prediction Model
with open("model/best_rf_model.pkl", 'rb') as file:
    liver_model = pickle.load(file)

# Load Liver Fibrosis Prediction Models
MODEL_PATH_LSTM = "model/lstm_model (1).h5"
MODEL_PATH_CNN = "model/cnn_model.h5"  # Add your CNN model path
MODEL_PATH_GAN = "model/classification_fnn_model.h5"  # Add your GAN model path

fibrosis_model_lstm = load_model(MODEL_PATH_LSTM)
fibrosis_model_cnn = load_model(MODEL_PATH_CNN)
fibrosis_model_gan = load_model(MODEL_PATH_GAN)

# Liver Fibrosis Class Labels
CLASSES = ['F0', 'F3', 'F4', 'F2', 'F1']

# Home Page Route
@app.route('/')
def index():
    return render_template('index.html')

# Liver Disease Prediction Route
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Handle form inputs for liver disease prediction
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

        # Prepare data for liver disease prediction
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
        prediction_proba = liver_model.predict_proba(input_data)
        result = "Pasien Terkena Penyakit Liver" if prediction_proba[0][1] >= 0.7 else "Pasien Tidak Terkena Penyakit Liver"

        return jsonify(prediction=result)

    return render_template('predict.html')  # Form for prediction

# Liver Fibrosis Prediction Route
@app.route('/fibrosis_predict', methods=['GET', 'POST'])
def fibrosis_predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join('uploads', filename)
            file.save(filepath)

            # Preprocess the image
            image = Image.open(filepath).convert('L')  # Convert to grayscale
            image = image.resize((64, 64))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0) / 255.0  # Normalize

            # Select model
            selected_model = request.form.get('model')  # Model selection input
            if selected_model == 'CNN':
                model = fibrosis_model_cnn
            elif selected_model == 'GAN':
                # GAN typically takes a noise vector as input, not an image, but here we are assuming it is adapted for classification
                image = image.reshape(1, -1)  # Flatten the image to match expected input shape
                model = fibrosis_model_gan
            elif selected_model == 'LSTM':
                model = fibrosis_model_lstm
            else:
                return "Model not selected or invalid"

            # Make prediction
            prediction = model.predict(image)
            confidence = np.max(prediction) * 100  # Confidence percentage
            predicted_class = CLASSES[np.argmax(prediction)]

            # Encode the image in base64 for displaying on the webpage
            with open(filepath, "rb") as img_file:
                encoded_img = base64.b64encode(img_file.read()).decode('utf-8')

            return render_template(
                'fibrosis_predict.html',
                prediction=f"{predicted_class} ({confidence:.2f}% confidence)",
                selected_model=selected_model,
                image_data=encoded_img
            )

    return render_template('fibrosis_predict.html', prediction=None, selected_model=None, image_data=None)

# Additional Routes
@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/konsultasi')
def konsultasi():
    return render_template('konsultasi.html')

@app.route('/gambar')
def gambar():
    return render_template('gambar.html')

@app.route('/profil.html')
def profile():
    return render_template('profil.html')

if __name__ == "__main__":
    # Create uploads folder if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
