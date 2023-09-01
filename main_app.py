# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 23:21:14 2023

@author: abir
"""

import pickle
import streamlit as st
import sys

sys.path.insert(1, 'C:/Users/91771/anaconda3/envs/newenv/Lib/site-packages/streamlit_option_menu')

from streamlit_option_menu import option_menu

diab_model = pickle.load(open('saved_models/diab_model.sav', 'rb'))
heart_model = pickle.load(open("saved_models/heart_model.sav",'rb'))
parkinsons_model = pickle.load(open("saved_models/parkinsons_model.sav", 'rb'))

st.title("Disease Prediction App")

st.markdown("""
        <p style="font-size: 10px;">📱 <strong>Mobile Users:</strong> Click the <strong>top left</strrong> icon to access sidebar content for instructions on using this app.</p>
        """,unsafe_allow_html=True)

st.sidebar.title("📋 Details")

st.sidebar.markdown("""
        Predict the likelihood of _Diabetes_, _Heart disease_, and _Parkinson's_ disease using machine learning models in this Streamlit web application. Take the help of   _**"How to Enter Data"**_  expander to see how to enter data. Also view the data sources down below.
    """)



with st.sidebar:
    st.markdown("Choose your preferred prediction from below here:")
    selected = option_menu('Multiple Disease Prediction System',

                          ['Diabetes Prediction',
                           'Heart Disease Prediction',
                           'Parkinsons Prediction'],
                          icons=['activity','heart','person'],
                          default_index=0)

st.sidebar.markdown(
        """
        ---
        ## Data Sources 📊

        - Diabetes Data: [Click Me](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
        - Heart Disease Data: [Click Me](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
        - Parkinson's Data: [Click Me](https://www.kaggle.com/datasets/debasisdotcom/parkinson-disease-detection)
        """
    )

sidebar_expander = st.sidebar.expander("📝 How to Enter Data")
with sidebar_expander:
        st.markdown(
            """
            To enter data for predictions, use the main page input boxes. Here are some sample examples:

            ### Diabetes Prediction
            - **Pregnancies**: 6
            - **Glucose**: 148
            - **BloodPressure**: 72
            - **SkinThickness**: 35
            - **Insulin**: 0
            - **BMI**: 33.6
            - **DiabetesPedigreeFunction**: 0.627
            - **Age**: 50


            ### Heart Disease Prediction
            - **Age**: 63
            - **Sex**: 1
            - **Chest Pain Type (cp)**: 3
            - **Resting Blood Pressure (trestbps)**: 145
            - **Cholesterol (chol)**: 233
            - **Fasting Blood Sugar (fbs)**: 1
            - **Resting ECG (restecg)**: 0
            - **Maximum Heart Rate (thalach)**: 150
            - **Exercise Induced Angina (exang)**: 0
            - **ST Depression Induced by Exercise (oldpeak)**: 2.3
            - **Slope of the Peak Exercise ST Segment (slope)**: 0
            - **Number of Major Vessels (ca)**: 0
            - **Thalassemia Type (thal)**: 1

            ### Parkinson's Prediction
            - **MDVP: Fo (Hz)**: 119.99200
            - **MDVP: Fhi (Hz)**: 157.30200
            - **MDVP: Flo (Hz)**: 74.99700
            - **MDVP: Jitter (%)**: 0.00784
            - **MDVP: Jitter (Abs)**: 0.00007
            - **MDVP: RAP**: 0.00370
            - **MDVP: PPQ**: 0.00554
            - **Jitter: DDP**: 0.01109
            - **MDVP: Shimmer**: 0.04374
            - **MDVP: Shimmer (dB)**: 0.42600
            - **Shimmer: APQ3**: 0.02182
            - **Shimmer: APQ5**: 0.03130
            - **MDVP: APQ**: 0.02971
            - **Shimmer: DDA**: 0.06545
            - **NHR**: 0.02211
            - **HNR**: 21.03300
            - **RPDE**: 0.414783
            - **DFA**: 0.815285
            - **Spread1**: -4.813031
            - **Spread2**: 0.266482
            - **D2**: 2.301442
            - **PPE**: 0.284654

            """
        )

st.sidebar.markdown(
        """
        ---

        ## Developed by
        👨‍💻 Abir Maiti

        📊 Data Analyst

        🇮🇳 India

        """
)


st.markdown("---")


# Diabetes Prediction Page
if (selected == 'Diabetes Prediction'):

    # page title
    st.title('Diabetes Prediction using Machine Learning')


    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')

    with col2:
        Glucose = st.text_input('Glucose Level')

    with col3:
        BloodPressure = st.text_input('Blood Pressure value')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value')

    with col2:
        Insulin = st.text_input('Insulin Level')

    with col3:
        BMI = st.text_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

    with col2:
        Age = st.text_input('Age of the Person')


    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction

    if st.button('Diabetes Test Result'):
        diab_prediction = diab_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

        if (diab_prediction[0] == 1):
          diab_diagnosis = "🩸 **Diagnosis:** The person has been **diagnosed with diabetes**."
        else:
          diab_diagnosis = "🥦 **Diagnosis:** The person does **not** have diabetes."

    st.success(diab_diagnosis)


# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):

    # page title
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')

    with col2:
        sex = st.text_input('Sex: 0=Female; 1=Male')

    with col3:
        cp = st.text_input('Chest Pain types')

    with col1:
        trestbps = st.text_input('Resting Blood Pressure')

    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')

    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')

    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')

    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')

    with col3:
        exang = st.text_input('Exercise Induced Angina')

    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')

    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')

    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')

    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    try:
        age = float(age)
        sex = int(sex)
        cp = float(cp)
        trestbps = float(trestbps)
        chol = float(chol)
        fbs = float(fbs)
        restecg = float(restecg)
        thalach = float(thalach)
        exang = float(exang)
        oldpeak = float(oldpeak)
        slope = float(slope)
        ca = float(ca)
        thal = float(thal)
    except ValueError:
        st.error("Please enter valid numeric values.")



    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction

    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]])

        if (heart_prediction[0] == 1):
          heart_diagnosis = "❤️ **Diagnosis:** The person has **heart disease**."
        else:
          heart_diagnosis = "💙 **Diagnosis:** The person does **not** have heart disease."


    st.success(heart_diagnosis)


# Parkinson's Prediction Page
if (selected == "Parkinsons Prediction"):

    # page title
    st.title("Parkinson's Disease Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')

    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')

    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')

    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')

    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

    with col1:
        RAP = st.text_input('MDVP:RAP')

    with col2:
        PPQ = st.text_input('MDVP:PPQ')

    with col3:
        DDP = st.text_input('Jitter:DDP')

    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')

    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')

    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')

    with col3:
        APQ = st.text_input('MDVP:APQ')

    with col4:
        DDA = st.text_input('Shimmer:DDA')

    with col5:
        NHR = st.text_input('NHR')

    with col1:
        HNR = st.text_input('HNR')

    with col2:
        RPDE = st.text_input('RPDE')

    with col3:
        DFA = st.text_input('DFA')

    with col4:
        spread1 = st.text_input('spread1')

    with col5:
        spread2 = st.text_input('spread2')

    with col1:
        D2 = st.text_input('D2')

    with col2:
        PPE = st.text_input('PPE')



    # code for Prediction
    parkinsons_diagnosis = ''

    # creating a button for Prediction
    if st.button("Parkinson's Test Result"):
        parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])

        if (parkinsons_prediction[0] == 1):
          parkinsons_diagnosis = "🎙️ **Diagnosis:** The person has been **diagnosed with Parkinson's disease**."
        else:
          parkinsons_diagnosis = "🚀 **Diagnosis:** The person does **not** have Parkinson's disease."

    st.success(parkinsons_diagnosis)
