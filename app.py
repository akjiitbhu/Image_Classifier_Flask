from __future__ import division, print_function
# coding=utf-8

import os
import numpy as np
import cv2
import base64
from PIL import Image
# Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# Flask utils
from flask import Flask, redirect, url_for, request, render_template, send_file
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


app = Flask(__name__)

print('Model loaded. Check http://127.0.0.1:5009/')

# Model saved with Keras model.save()
MODEL_PATH = 'models/w_unet.h5'

# Load your own trained model
model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')


def model_predict(img_path, model):
    image = cv2.imread(img_path, 1)
    image = cv2.resize(image, (256, 256))
    image = image/255.0
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    x = preprocess_input(image, mode='tf')
    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the image from post request
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        print("File Saved")
        preds = model_predict(file_path, model)
        n_result = preds > 0.5
        
        gray_image = np.reshape(np.array(n_result[0]*255).astype('uint8'),(256,256)) 
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)        
        processed_img = Image.fromarray(gray_image)
        processed_img.save('static/processed_image.jpeg')
        return render_template('index.html', image='processed_image.jpg')
        #result = "done"
        #return result
    return render_template('index.html')

if __name__ == '__main__':
      app.run(port=5009,debug=True)