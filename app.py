#http://localhost:8501/

import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier

model = CatBoostClassifier()
model.load_model('model/model.cbm')

data = pd.read_csv('datasets/dataset_diseases.csv', sep=';') 
desc = pd.read_csv('datasets/symptom_Description.csv', sep = ',')
prec = pd.read_csv('datasets/symptom_precaution.csv', sep = ',')

data = data.fillna('none')

def predict_disease(input_data):
    prediction = model.predict([input_data]) 
    return prediction[0]

st.title("Disease prediction model")
st.image('ai_assistent.jpeg',  use_column_width=True)
st.write("Enter symptoms and click 'Predict' for advice. Not a medical recommendation, consult a specialist")

symptom_1 = st.selectbox('Select the main symptom', data['Symptom_1'].unique())
symptom_2 = st.selectbox('Select a second symptom (if any)', data['Symptom_2'].unique())
symptom_3 = st.selectbox('Select a third symptom (if any)', data['Symptom_3'].unique())
symptom_4 = st.selectbox('Select a fourth symptom (if any)', data['Symptom_4'].unique())

input_data = pd.Series([symptom_1, symptom_2, symptom_3, symptom_4])

if st.button('Predict'):
    prediction = predict_disease(input_data)
    
    selected_desc = desc[desc['Disease'] == prediction[0]]['Description'].values[0]
    selected_prec = prec[prec['Disease'] == prediction[0]].iloc[:, 1:].values[0]

    st.write(f'Predicted disease: {prediction[0]}')
    st.write(f'Description: {selected_desc}')
    st.write('Recommendations:')
    for i, precaution in enumerate(selected_prec):
        st.write(f'Precaution_{i+1}: {precaution}')


