from flask import Flask, render_template,request
import sys

import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
import os


# Flask constructor takes the name of current module (__name__) as argument.
app = Flask(__name__)

model = tf.keras.models.load_model("resv") 

# The route() function of the Flask class is a decorator, which tells the application which URL should call the associated function.
@app.route("/")
def firstpage():
    return render_template("index.html")

#routes to predict
@app.route("/predict",methods=["GET","POST"])
def predict():
   
    print(request.files)
    image = request.files['img']
    print(image)
    image.save("img.jpg")

    #load the image
    my_image = load_img('img.jpg', target_size=(224, 224))

    #preprocess the image
    my_image = img_to_array(my_image)
    my_image = my_image.reshape((1, my_image.shape[0], my_image.shape[1], my_image.shape[2]))
    my_image = tf.keras.applications.vgg16.preprocess_input(my_image)
    my_image = my_image/255.0

    #make the prediction
    labels = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']
    prediction = np.argmax(model.predict(my_image))

    print(prediction)
    
    #   Putting prediction on output.html page  
    return render_template("index.html",output=labels[prediction])


# main driver function
if __name__=="__main__":
    app.run()
