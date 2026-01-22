import pickle
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import os

app = FastAPI()

# Load model and scaler
ridge_path = os.path.join('model', 'ridge.pkl')
scaler_path = os.path.join('model', 'scaler.pkl')
ridge_model = None
standard_model = None
if os.path.exists(ridge_path) and os.path.exists(scaler_path):
    ridge_model = pickle.load(open(ridge_path, 'rb'))
    standard_model = pickle.load(open(scaler_path, 'rb'))


@app.get('/', response_class=HTMLResponse)
async def index():
    if os.path.exists('templates/index.html'):
        with open('templates/index.html', 'r', encoding='utf-8') as f:
            return f.read()
    return HTMLResponse('<h1>Diabetic Model</h1><p>No index.html found</p>')


@app.post('/diabetes', response_class=HTMLResponse)
async def diabetes(request: Request):
    if ridge_model is None or standard_model is None:
        return HTMLResponse('<h1>Model files missing</h1>', status_code=500)

    form = await request.form()
    try:
        Glucose = float(form.get('Glucose', 0))
        BloodPressure = float(form.get('BloodPressure', 0))
        SkinThickness = float(form.get('SkinThickness', 0))
        Insulin = float(form.get('Insulin', 0))
        BMI = float(form.get('BMI', 0))
        Age = float(form.get('Age', 0))
        Weight = float(form.get('Weight', 0))
        Height = float(form.get('Height', 0))

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

        if os.path.exists('templates/Predicted.html'):
            with open('templates/Predicted.html', 'r', encoding='utf-8') as f:
                html = f.read()
            html = html.replace('{{results}}', str(final_result))
            return HTMLResponse(html)

        return HTMLResponse(str(final_result))
    except Exception as e:
        return HTMLResponse(f'<h1>Error: {e}</h1>', status_code=400)
