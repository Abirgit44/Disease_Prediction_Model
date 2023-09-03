# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 23:21:14 2023

@author: abir
"""

import pickle
import streamlit as st
import sys
from sklearn.metrics import accuracy_score

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


# Create a sidebar expander
sidebar_expander = st.sidebar.expander("📝 How to Enter Data")

# Define a dictionary to store input fields for each disease
input_fields = {
    "Diabetes Prediction": """
        To enter data for diabetes prediction, here are instructions based on **World Health Organization**:

        - **Pregnancies**: Enter the number of times pregnant (whole number, e.g., 6).
          - Normal Numeric Range: [0, 20]
          - [Read More](https://en.wikipedia.org/wiki/Gestational_diabetes)

        - **Glucose**: Enter the patient's fasting blood glucose concentration (mg/dL, e.g., 148).
          - Normal Numeric Range: [70, 130] mg/dL
          - [Read More](https://en.wikipedia.org/wiki/Glucose)

        - **BloodPressure(Diastolic BP)**: Enter the diastolic blood pressure (mm Hg, e.g., 72).
          - Normal Numeric Range: [60, 80] mm Hg
          - [Read More](https://en.wikipedia.org/wiki/Blood_pressure)

        - **SkinThickness**: Enter the patient's skin thickness (mm, e.g., 35).
          - Normal Numeric Range: [10, 45] mm
          - [Read More](https://en.wikipedia.org/wiki/Human_skin)

        - **Insulin**: Enter the patient's insulin level (mu U/ml, e.g., 0).
          - Normal Numeric Range: [0, 35] mu U/ml
          - [Read More](https://en.wikipedia.org/wiki/Insulin)

        - **BMI**: Enter the patient's body mass index (kg/m², e.g., 33.6).
          - Normal Numeric Range: [18.5, 24.9] kg/m²
          - [Read More](https://en.wikipedia.org/wiki/Body_mass_index)

        - **DiabetesPedigreeFunction**: Enter the diabetes pedigree function (e.g., 0.627).
          - Normal Numeric Range: [0.0, 1.6]
          - [Read More](https://en.wikipedia.org/wiki/Diabetes_pedigree_function)

        - **Age**: Enter the patient's age (whole number, e.g., 50).
          - Normal Numeric Range: [20, 60] years
          - [Read More](https://en.wikipedia.org/wiki/Ageing)
        """,
    "Heart Disease Prediction": """
        To enter data for heart disease prediction, here are instructions based on **World Health Organization**:

        - **Age**: Enter the patient's age.
          - Normal Numeric Range: [0, 130] years
          - [Read More](https://en.wikipedia.org/wiki/Ageing)

        - **Sex**: Enter the patient's sex (0 for female, 1 for male).
          - [Read More](https://en.wikipedia.org/wiki/Human_sex_ratio)

        - **Chest Pain Type (cp)**: Enter the type of chest pain.
          - Normal Numeric Range: [0, 3]
          - [Read More](https://en.wikipedia.org/wiki/Chest_pain)

        - **Resting Blood Pressure(Diastolic BP)**: Enter the resting diastolic blood pressure.
          - Normal Numeric Range: [60, 90] mm Hg
          - [Read More](https://en.wikipedia.org/wiki/Blood_pressure)

        - **Cholesterol (chol)**: Enter the patient's cholesterol level.
          - Normal Numeric Range: [125, 200] mg/dL
          - [Read More](https://en.wikipedia.org/wiki/Cholesterol)

        - **Fasting Blood Sugar (> 120 mg/dl)**: Enter fasting blood sugar status (0 for False, 1 for True).
          - [Read More](https://en.wikipedia.org/wiki/Blood_sugar_level)

        - **Resting ECG (restecg)**: Enter the resting electrocardiographic results.
          - Normal Numeric Range: [0, 2]
          - [Read More](https://en.wikipedia.org/wiki/Electrocardiography)

        - **Maximum Heart Rate (thalach)**: Enter the patient's maximum heart rate.
          - Normal Numeric Range: [60, 100] bpm
          - [Read More](https://en.wikipedia.org/wiki/Heart_rate)

        - **Exercise Induced Angina (exang)**: Enter exercise-induced angina status (0 for No, 1 for Yes).
          - [Read More](https://en.wikipedia.org/wiki/Angina)

        - **ST Depression Induced by Exercise (oldpeak)**: Enter ST depression induced by exercise.
          - Normal Numeric Range: [0.0, 4.0]
          - [Read More](https://en.wikipedia.org/wiki/ST_depression)

        - **Slope of the Peak Exercise ST Segment (slope)**: Enter the slope of the peak exercise ST segment.
          - Normal Numeric Range: [0, 2]
          - [Read More](https://en.wikipedia.org/wiki/ST_segment)

        - **Number of Major Vessels (ca)**: Enter the number of major vessels colored by fluoroscopy.
          - Normal Numeric Range: [0, 3]
          - [Read More](https://en.wikipedia.org/wiki/Coronary_artery_disease)

        - **Thalassemia Type (thal)**: Enter the type of thalassemia.
          - Normal Numeric Range: [0, 2]
          - [Read More](https://en.wikipedia.org/wiki/Thalassemia)
        """,
    "Parkinsons Prediction": """
        To enter data for Parkinson's prediction, here are instructions based on **World Health Organization**:

        - **MDVP: Fo (Hz)**: Enter the average vocal fundamental frequency in Hz.
          - Normal Numeric Range: [85, 260] Hz
          - [Read More](https://en.wikipedia.org/wiki/Fundamental_frequency)

        - **MDVP: Fhi (Hz)**: Enter the maximum vocal fundamental frequency in Hz.
          - Normal Numeric Range: [175, 280] Hz
          - [Read More](https://en.wikipedia.org/wiki/Fundamental_frequency)

        - **MDVP: Flo (Hz)**: Enter the minimum vocal fundamental frequency in Hz.
          - Normal Numeric Range: [85, 260] Hz
          - [Read More](https://en.wikipedia.org/wiki/Fundamental_frequency)

        - **MDVP: Jitter (%)**: Enter the percentage variation in fundamental frequency.
          - Normal Numeric Range: [0.0, 0.1]
          - [Read More](https://en.wikipedia.org/wiki/Jitter)

        - **MDVP: Jitter (Abs)**: Enter the absolute variation in fundamental frequency.
          - Normal Numeric Range: [0.0, 0.001]
          - [Read More](https://en.wikipedia.org/wiki/Jitter)

        - **MDVP: RAP**: Enter the relative average perturbation in fundamental frequency.
          - Normal Numeric Range: [0.0, 0.02]
          - [Read More](https://en.wikipedia.org/wiki/RAP)

        - **MDVP: PPQ**: Enter the percentage of perturbations in fundamental frequency.
          - Normal Numeric Range: [0.0, 0.02]
          - [Read More](https://en.wikipedia.org/wiki/PPQ)

        - **Jitter: DDP**: Enter the absolute difference between the average absolute difference of consecutive periods.
          - Normal Numeric Range: [0.0, 0.05]
          - [Read More](https://en.wikipedia.org/wiki/Jitter)

        - **MDVP: Shimmer**: Enter the variation in amplitude.
          - Normal Numeric Range: [0.0, 0.1]
          - [Read More](https://en.wikipedia.org/wiki/Shimmer_(acoustics))

        - **MDVP: Shimmer (dB)**: Enter the variation in amplitude in decibels.
          - Normal Numeric Range: [0.0, 1.0]
          - [Read More](https://en.wikipedia.org/wiki/Shimmer_(acoustics))

        - **Shimmer: APQ3**: Enter the amplitude perturbation quotient.
          - Normal Numeric Range: [0.0, 0.05]
          - [Read More](https://en.wikipedia.org/wiki/Shimmer_(acoustics))

        - **Shimmer: APQ5**: Enter another amplitude perturbation quotient.
          - Normal Numeric Range: [0.0, 0.1]
          - [Read More](https://en.wikipedia.org/wiki/Shimmer_(acoustics))

        - **MDVP: APQ**: Enter a different measure of amplitude perturbation quotient.
          - Normal Numeric Range: [0.0, 0.1]
          - [Read More](https://en.wikipedia.org/wiki/APQ)

        - **Shimmer: DDA**: Enter the difference between the amplitudes of consecutive periods.
          - Normal Numeric Range: [0.0, 0.2]
          - [Read More](https://en.wikipedia.org/wiki/Shimmer_(acoustics))

        - **NHR**: Enter the noise-to-harmonics ratio.
          - Normal Numeric Range: [0.0, 0.2]
          - [Read More](https://en.wikipedia.org/wiki/Noise-to-harmonics_ratio)

        - **HNR**: Enter the harmonics-to-noise ratio.
          - Normal Numeric Range: [0.0, 1.0]
          - [Read More](https://en.wikipedia.org/wiki/Harmonics-to-noise_ratio)

        - **RPDE**: Enter a nonlinear dynamical complexity measure.
          - Normal Numeric Range: [0.0, 0.5]
          - [Read More](https://en.wikipedia.org/wiki/Poincar%C3%A9_plot)

        - **DFA**: Enter the signal fractal scaling exponent.
          - Normal Numeric Range: [0.0, 2.0]
          - [Read More](https://en.wikipedia.org/wiki/Detrended_fluctuation_analysis)

        - **spread1**: Enter a nonlinear measure of fundamental frequency variation.
          - Normal Numeric Range: [0.0, 1.0]
          - [Read More](https://en.wikipedia.org/wiki/Vocal_fold)

        - **spread2**: Enter another nonlinear measure of fundamental frequency variation.
          - Normal Numeric Range: [0.0, 1.0]
          - [Read More](https://en.wikipedia.org/wiki/Vocal_fold)

        - **PPE**: Enter another nonlinear measure of fundamental frequency variation.
          - Normal Numeric Range: [0.0, 0.8]
          - [Read More](https://en.wikipedia.org/wiki/Vocal_fold)
        """
}

# Display input fields within the sidebar expander based on the selected disease
sidebar_expander.markdown(input_fields[selected])




st.sidebar.markdown(
        """
        ---

        ## Developed by
        👨‍💻 Abir Maiti

        📊 Data Analyst

        🇮🇳 India

        [![GitHub](https://img.shields.io/badge/GitHub-Profile-blue?logo=github)](https://bit.ly/Abirgit44) [![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin)](https://bit.ly/linkAbir)
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
        try:
                    Pregnancies = float(Pregnancies)
                    Glucose = float(Glucose)
                    BloodPressure = float(BloodPressure)
                    SkinThickness = float(SkinThickness)
                    Insulin = float(Insulin)
                    BMI = float(BMI)
                    DiabetesPedigreeFunction = float(DiabetesPedigreeFunction)
                    Age = float(Age)

                    if any(x < 0 for x in [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]):
                        st.error("Please enter non-negative values for all input fields.")
                    else:
                        diab_prediction = diab_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
                        if (diab_prediction[0] == 1):
                              diab_diagnosis = "🩸 **Diagnosis:** The person has been classified as a patient **with diabetes**."
                        else:
                              diab_diagnosis = "🥦 **Diagnosis:** The person has been classified as a patient to **not** have diabetes."

                        st.success(diab_diagnosis)

        except ValueError:
            st.error("Please enter valid numeric values for Diabetes Prediction.")

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
        thal = st.text_input('Thalassemia:0 = normal; 1 = fixed defect; 2 = reversable defect')


    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction

    if st.button('Heart Disease Test Result'):
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

            if any(x < 0 for x in [age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]):
                st.error("Please enter non-negative values for all input fields.")
            else:
                    heart_prediction = heart_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]])

                    if (heart_prediction[0] == 1):
                         heart_diagnosis = "❤️ **Diagnosis:** The person has been classified as a patient **with heart disease**."
                    else:
                         heart_diagnosis = "💙 **Diagnosis:** The person has been classified as a patient to **not** have heart disease."

                    st.success(heart_diagnosis)
        except ValueError:
             st.error("Please enter valid numeric values for Heart Disease Prediction.")

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
        try:
            fo = float(fo)
            fhi = float(fhi)
            flo = float(flo)
            Jitter_percent = float(Jitter_percent)
            Jitter_Abs = float(Jitter_Abs)
            RAP = float(RAP)
            PPQ = float(PPQ)
            DDP = float(DDP)
            Shimmer = float(Shimmer)
            Shimmer_dB = float(Shimmer_dB)
            APQ3 = float(APQ3)
            APQ5 = float(APQ5)
            APQ = float(APQ)
            DDA = float(DDA)
            NHR = float(NHR)
            HNR = float(HNR)
            RPDE = float(RPDE)
            DFA = float(DFA)
            spread1 = float(spread1)
            spread2 = float(spread2)
            D2 = float(D2)
            PPE = float(PPE)

            if any(x < 0 for x in [fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]):

                        st.error("Please enter non-negative values for all input fields.")
            else:
                parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])

                if (parkinsons_prediction[0] == 1):
                  parkinsons_diagnosis = "🎙️ **Diagnosis:** The person has been classified as a patient **with Parkinson's disease**."
                else:
                  parkinsons_diagnosis = "🚀 **Diagnosis:** The person has been classified as a patient to **not** have Parkinson's disease."

                st.success(parkinsons_diagnosis)
        except ValueError:
              st.error("Please enter valid numeric values for Parkinson's Disease Prediction.")
