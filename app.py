from flask import Flask,request,send_from_directory
from flask import render_template
from keras.datasets import mnist
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from sklearn.utils import shuffle

# Recreate the exact same model, including its weights and the optimizer
model = tf.keras.models.load_model(r'C:\Users\Shashi\Downloads\ocr\model.h5')

# Show the model architecture
model.summary()

word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}

COUNT=0
app = Flask(__name__)
 

@app.route("/",methods=["GET","POST"])
def guess():
    return render_template("Home.html")
@app.route("/hcr",methods=["GET","POST"])
def about():
    return render_template("hcr.html")
@app.route("/contact",methods=["GET","POST"])
def contact():
    return render_template("Contact.html")
@app.route("/picture")
def picture():
    return send_from_directory('database',"0.jpg")

@app.route("/predict",methods=["POST"])
def predict():
        global COUNT
        img=request.files['image']
        img.save('database/{}.jpg'.format(COUNT))
        img = cv2.imread(r'C:\Users\Shashi\Downloads\ocr\database\0.jpg')
        img_copy = img.copy()
        img_copy = cv2.GaussianBlur(img_copy, (7,7), 0)
        img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        _, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)
        img_final = cv2.resize(img_thresh, (28,28))
        img_final =np.reshape(img_final, (1,28,28,1))
        img_pred = word_dict[np.argmax(model.predict(img_final))]
        return render_template("output.html",name=img_pred)
        
if __name__ =="__main__":
   app.run(port=12000,debug=True)