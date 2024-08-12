from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import json

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers=["*"],
)

class diabetes_input(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int
    
# Loading the saved model
diabetes_model = pickle.load(open('diabetes_model.sav','rb'))
diabetes_scaler = pickle.load(open('diabetes_scaler.sav', 'rb'))

@app.post('/diabetes_prediction')
def diabetes_pred(input_parameters: diabetes_input):
    input_data = input_parameters.dict()  # Convert to dictionary directly

    pregnancies = input_data['Pregnancies']
    glucose = input_data['Glucose']
    bloodpressure = input_data['BloodPressure']
    skinthickness = input_data['SkinThickness']
    insulin = input_data['Insulin']
    bmi = input_data['BMI']
    diabetespedigreefunction = input_data['DiabetesPedigreeFunction']
    age = input_data['Age']

    input_list = [pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigreefunction, age]

    # Scale the input data
    scaled_input = diabetes_scaler.transform([input_list])

    # Make prediction
    prediction = diabetes_model.predict(scaled_input)

    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'