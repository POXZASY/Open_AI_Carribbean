import csv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

from pathlib import Path

labels = ["concrete_cement", "healthy_metal", "incomplete", "irregular_metal", "other"]

IMG_WIDTH = 224
IMG_HEIGHT = 224

#load in the model
model = load_model('model.h5')
#check this is the same as model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


with open("predictions.csv", "w", newline = '') as file:
    writer = csv.writer(file)
    writer.writerow(["id", "concrete_cement", "healthy_metal", "incomplete", "irregular_metal", "other"])
    #iterate through every image
    path = Path('./prediction_images').glob('*.png')
    for p in path:
        filename = str(p)
        print(filename)
        # predicting images
        img = image.load_img(filename, target_size=(IMG_WIDTH, IMG_HEIGHT))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = model.predict(images, batch_size=10)
        y_classes = classes.argmax(axis=-1)
        loc = y_classes[0]
        print("Prediction: " + labels[loc]+ " Probability: "+str(classes[0][loc]))
        idval = (filename.split('\\'))[1]
        idval = idval.split('.')[0]
        values = []
        for i in range(5):
            values.append("{:.4f}".format(float(classes[0][i])))
        writer.writerow([idval, values[0], values[1], values[2], values[3], values[4]])
    

    
