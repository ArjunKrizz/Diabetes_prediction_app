import streamlit as st
import pickle
import numpy as np

# Load the saved model from file
with open('decision_tree_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Function for prediction
def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    prediction = loaded_model.predict(input_data)
    return prediction

# Streamlit app with HTML/CSS styling
st.title('Diabetes Prediction')
st.markdown("""
<style>
.big-font {
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# Input fields for user to input values
st.markdown('<p class="big-font">Enter Patient Details:</p>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, step=1)
    glucose = st.number_input('Glucose', min_value=0, max_value=200, step=1)
    blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=150, step=1)
    skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, step=1)

with col2:
    insulin = st.number_input('Insulin', min_value=0, max_value=900, step=1)
    bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, step=0.1)
    diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, step=0.01)
    age = st.number_input('Age', min_value=0, max_value=120, step=1)

if st.button('Predict', key='prediction'):
    prediction = predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age)
    if prediction[0] == 0:
        st.success('No Diabetes')
    else:
        st.error('Diabetes Detected')
