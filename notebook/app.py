import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from src.exception import CustomException


# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("notebook\model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("a.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    if request.method == 'POST':
        file1 = request.files['file']
    else:
        raise BadRequestError('No file submitted')
    file2=pd.read_csv('file1')
    file3 = file2.reshape(file2.shape[0], file2.shape[1], 1)
    prediction = model.predict(file2.shape[1],1)
    y_pred1=[]
    for i in prediction:
        if i <= 0.5:
            y_pred1.append(0)
        else:
            y_pred1.append(1)
    return render_template("a.html", prediction_text = "The customer is {}".format(y_pred1))

if __name__ == "__main__":
    flask_app.run(host="0.0.0.0" , port =5000)