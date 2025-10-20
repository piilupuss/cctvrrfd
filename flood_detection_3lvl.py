import cv2
import numpy as np
import tensorflow as tf
#from tensorflow import keras
#from keras.models import Sequential, load_models
from tensorflow.python.keras.models import load_model
import os
#from my_function import coral_loss
from keras._tf_keras.keras.preprocessing import image

# Compile model with CORAL loss
def coral_loss(y_true, y_pred):
    # Convert to float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Compute the mean of the true and predicted values
    y_true_mean = tf.reduce_mean(y_true, axis=0)
    y_pred_mean = tf.reduce_mean(y_pred, axis=0)

    # Center the true and predicted values
    y_true_centered = y_true - y_true_mean
    y_pred_centered = y_pred - y_pred_mean

    # Compute the covariance matrices
    cov_true = tf.matmul(y_true_centered, y_true_centered, transpose_a=True) / tf.cast(tf.shape(y_true)[0], tf.float32)
    cov_pred = tf.matmul(y_pred_centered, y_pred_centered, transpose_a=True) / tf.cast(tf.shape(y_pred)[0], tf.float32)

    # Compute the CORAL loss
    loss = tf.reduce_mean(tf.square(cov_true - cov_pred))
    return loss 

def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resize to fit MobileNet input
    img = img / 255.0  # Normalize
    return np.expand_dims(img, axis=0)

def detect_flood(model, image_folder):
    results = {}
    for image_file in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_file)
        img = load_and_preprocess_image(image_path)
        prediction = model.predict(img)
        results[image_file] = 'Flood' if prediction[0][0] > 0.5 else 'No Flood'
    return results

# Define a function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to match model input
    img_array = image.img_to_array(img) / 255.0  # Convert to array and normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Define a function to classify the image
def classify_image(img_path):
    # file_model = "/Users/nagailab/Documents/Dev/water_level_classifier/water_level_classifier_stage2_3.h5"
    file_model = "/Users/nagailab/Documents/Dev/water_level_classifier/water_level_classifier_stage2_20250830.keras"
    model = tf.keras.models.load_model(file_model, custom_objects={'coral_loss': coral_loss})
    model.compile()
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    # return class_idx, predictions[0]
    # return class_idx
    if class_idx == 0:
        hasil = "Dry"
    elif class_idx == 1:
        hasil = "Wet"
    else:
        hasil = "Flood" 
    return hasil

if __name__ == "__main__":
    #model = load_model('models/mobilenetv2.h5')
    # model = load_model('models/flood_detector.h5')
    model = tf.keras.models.load_model('water_level_classifier_stage2_3.h5', custom_objects={'coral_loss': coral_loss})
    # model = tf.keras.models.load_model('water_level_classifier_stage2_20250830.keras', custom_objects={'coral_loss': coral_loss})
    #tf.keras.applications.MobileNetV2(input_shape=(height, width, channels), include_top=False, weights='imagenet')
    results = detect_flood(model, 'data/captured_images')
    # print(results)
