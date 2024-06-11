from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

UPLOAD_FOLDER = 'uploads'
MODELS_FOLDER = 'models'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/lung_infection_detection', methods=['GET', 'POST'])
def lung_infection_detection():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('lung_upload.html', message='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('lung_upload.html', message='No selected file')
        if file and allowed_file(file.filename):
            if not os.path.exists(UPLOAD_FOLDER):
                os.makedirs(UPLOAD_FOLDER)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            return render_template('lung_options.html', filename=file.filename)
    return render_template('lung_upload.html')

@app.route('/predict_cnn/<filename>')
def predict_cnn(filename):
    # Implement CNN prediction logic
    print(filename)

    # Load the trained model
    model = load_model('models/covid_cnn_model.h5')

    class_names = ["Normal","Non-Covid","Covid"]
    interpreted_message = ["You do not have any Lung infection!! Hip Hip Hurray!!",
                           "You have Non-Covid Lung Infection. Please consult Doctors.",
                           "You have Covid Infection. Please maintian Quarantine."]
    
    # Process the image and make predictions
    image_size = (64,64)
    img_data = cv2.imread(os.path.join('uploads', filename))
    img_data = cv2.resize(img_data.copy(), image_size,interpolation=cv2.INTER_AREA)
    img_data = img_data/255

    prediction = model.predict(np.array([img_data]))
    predicted_class = class_names[np.argmax(prediction)]
    predicted_message = interpreted_message[np.argmax(prediction)]
    return render_template('result.html', filename=filename, prediction=predicted_message)


@app.route('/predict_dnn/<filename>')
def predict_dnn(filename):
    # Implement DNN prediction logic
    print(filename)
    model = load_model('models/covid_dnn_model.h5')

    class_names = ["Normal","Non-Covid","Covid"]
    interpreted_message = ["You do not have any Lung infection!! Hip Hip Hurray!!",
                           "You have Non-Covid Lung Infection. Please consult Doctors.",
                           "You have Covid Infection. Please maintian Quarantine."]

    image_size = (64,64)
    img_data = cv2.imread(os.path.join('uploads', filename))
    img_data = cv2.resize(img_data.copy(), image_size,interpolation=cv2.INTER_AREA)
    img_data = img_data/255

    prediction = model.predict(np.array([img_data]))
    predicted_class = class_names[np.argmax(prediction[0])]
    predicted_message = interpreted_message[np.argmax(prediction)]
    return render_template('result.html', filename=filename, prediction=predicted_message)


if __name__ == '__main__':
    app.run(debug=True)

