import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("notebook\model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("Model_UI.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    if 'file' in request.files:
        file1 = request.files['file']
    else:
        raise BadRequestError('No file submitted')
    
    file2 = file1.reshape(file1.shape[0], file1.shape[1],1)
    
    prediction = model.predict(file2.shape[1],1)
    y_pred1=[]
    for i in prediction:
        if i <= 0.5:
            y_pred1.append(0)
        else:
            y_pred1.append(1)
    return render_template("Model_UI.html", prediction_text = "The customer is {}".format(y_pred1))

if __name__ == "__main__":
    flask_app.run(debug=True)