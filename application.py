import pickle
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI()

ridge_model = pickle.load(open('model/ridge.pkl', 'rb'))
standard_model = pickle.load(open('model/scaler.pkl', 'rb'))

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html", "r") as f:
        return f.read()

@app.post("/diabetes", response_class=HTMLResponse)
async def diabetes(request: Request):
    form_data = await request.form()
    
    try:
        Glucose = float(form_data.get('Glucose', 0))
        BloodPressure = float(form_data.get('BloodPressure', 0))
        SkinThickness = float(form_data.get('SkinThickness', 0))
        Insulin = float(form_data.get('Insulin', 0))
        BMI = float(form_data.get('BMI', 0))
        Age = float(form_data.get('Age', 0))
        Weight = float(form_data.get('Weight', 0))
        Height = float(form_data.get('Height', 0))

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
        
        with open("templates/Predicted.html", "r") as f:
            html_content = f.read()
        
        html_content = html_content.replace("{{results}}", str(final_result))
        return html_content
    
    except Exception as e:
        return f"<h1>Error: {str(e)}</h1>"
