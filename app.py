import os
import MySQLdb
from flask import Flask, session, url_for, redirect, render_template, request, abort, flash
 
 
from werkzeug.utils import secure_filename
import numpy as np
import joblib
import numpy as np
from flask import Flask, redirect, url_for, request, render_template
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from PIL import Image
from database import *

from pathlib import Path
 
app = Flask(__name__)
app.secret_key='detection'
 
app.config['UPLOAD_FOLDER'] = 'static/uploads'
 


 


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/registera")
def registera():
    return render_template("register.html")

@app.route("/logina")
def logina():
    return render_template("login.html")

@app.route("/predicta")
def predicta():
    return render_template("predict.html")


@app.route("/predictionoutputa")
def predictoutputa():
    return render_template("predictionoutput.html")


@app.route("/register",methods=['POST','GET'])
def signup():
    if request.method=='POST':
        username=request.form['username']
        email=request.form['email']
        password=request.form['password']
        status = user_reg(username,email,password)
        if status == 1:
            return render_template("/login.html")
        else:
            return render_template("/register.html",m1="failed")        
    

@app.route("/login",methods=['POST','GET'])
def login():
    if request.method=='POST':
        username=request.form['username']
        password=request.form['password']
        status = user_loginact(request.form['username'], request.form['password'])
        print(status)
        if status == 1:                                      
            return render_template("/menu.html", m1="sucess")
        else:
            return render_template("/login.html", m1="Login Failed")
             
app.static_folder = 'static'

    
# @app.route("/")
# def home():
#     return render_template("index.html")

@app.route('/logouta')
def logout():
    # Clear the session data
    session.clear()
    return redirect(url_for('logina'))
    
@app.route('/prediction1', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':

        # -------------------------------
        # SAVE UPLOADED IMAGE
        # -------------------------------
        image_file = request.files['image']
        filename = secure_filename(image_file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(file_path)

        # -------------------------------
        # LOAD LR MODEL & SCALER
        # -------------------------------
        lr_model = joblib.load('lr_model.pkl')
        scaler = joblib.load('scaler.pkl')

        # -------------------------------
        # CLASS LABELS
        # -------------------------------
        classes = {0: "stage1", 1: "stage2", 2: "thyroid"}

        # -------------------------------
        # IMAGE PREPROCESSING (ML STYLE)
        # -------------------------------
        image = cv2.imread(file_path)
        image = cv2.resize(image, (224, 224))
        image = image / 255.0
        image = image.flatten().reshape(1, -1)
        image = scaler.transform(image)

        # -------------------------------
        # PREDICTION
        # -------------------------------
        result = lr_model.predict(image)[0]
        res = classes[result]

        # -------------------------------
        # MEDICAL CONTENT
        # -------------------------------
        description = ""
        remedies = ""
        exercise_recommendations = ""
        diet_suggestions = ""
        precautions = ""
        food_avoidance = ""

        if res == "stage1":
            description = "Hypothyroidism caused by low thyroid hormone production, often due to Hashimoto's thyroiditis or iodine deficiency."
            remedies = "Consume iodine-rich foods, selenium-rich foods, and zinc-rich foods."
            exercise_recommendations = (
                "• Light walking (15–20 minutes daily)<br>"
                "• Gentle yoga and stretching<br>"
                "• Swimming<br>"
                "• Light resistance training<br>"
                "• Avoid high-intensity workouts initially"
            )
            diet_suggestions = (
                "• Iodine-rich foods<br>"
                "• Selenium sources<br>"
                "• Zinc sources<br>"
                "• Lean proteins<br>"
                "• Whole grains and vegetables"
            )
            precautions = (
                "• Monitor thyroid levels regularly<br>"
                "• Take medication as prescribed<br>"
                "• Reduce stress<br>"
                "• Maintain regular check-ups"
            )
            food_avoidance = (
                "• Raw cruciferous vegetables<br>"
                "• Excess soy<br>"
                "• Processed foods<br>"
                "• Excess caffeine and alcohol"
            )

        elif res == "stage2":
            description = "Moderate thyroid dysfunction with noticeable symptoms such as fatigue, weight gain, and mood changes."
            remedies = "Levothyroxine therapy under medical supervision and lifestyle modifications."
            exercise_recommendations = (
                "• Moderate walking (20–30 minutes)<br>"
                "• Swimming or cycling<br>"
                "• Resistance training 2–3 times/week<br>"
                "• Stress-reducing exercises"
            )
            diet_suggestions = (
                "• Anti-inflammatory foods<br>"
                "• High-fiber diet<br>"
                "• Vitamin D and B-rich foods<br>"
                "• Adequate protein intake"
            )
            precautions = (
                "• Strict medication adherence<br>"
                "• Regular blood tests<br>"
                "• Monitor heart rate<br>"
                "• Avoid extreme diets"
            )
            food_avoidance = (
                "• Excess sugar<br>"
                "• Trans fats<br>"
                "• Highly processed foods<br>"
                "• Excess soy products"
            )

        elif res == "thyroid":
            description = "Severe thyroid dysfunction requiring immediate medical attention."
            remedies = "Hospitalization, intensive medical management, and possible surgery or radioactive iodine therapy."
            exercise_recommendations = (
                "• Bed-level movement only<br>"
                "• Breathing exercises<br>"
                "• Passive stretching<br>"
                "• No strenuous activity"
            )
            diet_suggestions = (
                "• Soft foods<br>"
                "• Small frequent meals<br>"
                "• High-protein liquids<br>"
                "• Medically supervised nutrition"
            )
            precautions = (
                "• Immediate medical supervision<br>"
                "• Continuous monitoring<br>"
                "• Infection prevention<br>"
                "• No self-treatment"
            )
            food_avoidance = (
                "• Raw vegetables<br>"
                "• Hard-to-digest foods<br>"
                "• Alcohol<br>"
                "• High-fiber foods during crisis"
            )

        # -------------------------------
        # RENDER RESULT
        # -------------------------------
        return render_template(
            'predictionoutput.html',
            prediction={
                'p': res,
                'image': os.path.join(app.config['UPLOAD_FOLDER'], filename),
                'data': res
            },
            des=description,
            rem=remedies,
            exercise=exercise_recommendations,
            diet=diet_suggestions,
            prec=precautions,
            avoid=food_avoidance
        )
if __name__ == "__main__":
    app.run(debug=True)

     
     