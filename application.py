import pickle
from flask import Flask, request, render_template 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

application= Flask(__name__)
app=application

ridge_model = pickle.load(open('model/ridge.pkl','rb'))
standard_model=pickle.load(open('model/scaler.pkl','rb'))

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/diabetes', methods=['POST', 'GET'])
def diabetes():
    if request.method == 'POST':
        Glucose = float(request.form['Glucose'])
        BloodPressure = float(request.form['BloodPressure'])
        SkinThickness = float(request.form['SkinThickness'])
        Insulin = float(request.form['Insulin'])
        BMI = float(request.form['BMI'])
        Age = float(request.form['Age'])
        Weight = float(request.form['Weight'])
        Height = float(request.form['Height'])

        new_data_scaled = standard_model.transform([[Glucose, BloodPressure, SkinThickness, Insulin, BMI, Age, Weight, Height]])
        result = ridge_model.predict(new_data_scaled)

        ai_suggestions = {
            "Message": "Sorry You are Diabetic",
            
            "Medicines (Informational Only)": [
                "Metformin (commonly first-line treatment)",
                "Insulin (for advanced cases)",
                "Sulfonylureas (doctor prescribed)"
            ],


            "Lifestyle Advice": [
                "Avoid sugary and processed foods",
                "Daily 30 minutes exercise",
                "Monitor blood sugar regularly"
            ]


        }

        ai_suggestions2 = {
            "Message": "Congratulations! You are not Diabetic",

            "Prevention Tips": [
                "Maintain healthy diet",
                "Exercise regularly",
                "Avoid excessive sugar intake"
            ]

        }

        final_result = ai_suggestions if result[0] == 1 else ai_suggestions2

        return render_template("Predicted.html", results=final_result)
    else:
        return render_template("index.html")


if __name__=='__main__':
    app.run(host='0.0.0.0',port=8000)
