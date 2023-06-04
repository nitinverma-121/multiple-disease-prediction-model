import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# loading the saved models
dia_model= pickle.load(open('D:/Users/NITIN VERMA/Desktop/Code for basics ML/New Project/Diabetes/dia_model.pkl', 'rb'))
dia_scaler= pickle.load(open('D:/Users/NITIN VERMA/Desktop/Code for basics ML/New Project/Diabetes/dia_scaler.pkl', 'rb'))


par_model= pickle.load(open('D:/Users/NITIN VERMA/Desktop/Code for basics ML/New Project/parkinson disease/par_model.pkl', 'rb'))
par_scaler= pickle.load(open('D:/Users/NITIN VERMA/Desktop/Code for basics ML/New Project/parkinson disease/par_scaler.pkl', 'rb'))

heart_model= pickle.load(open('D:/Users/NITIN VERMA/Desktop/Code for basics ML/New Project/Heart disease/heart_model.pkl', 'rb'))


# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Multiple Disease Prediction System',       
                          ['Diabetes Prediction',
                           'Heart Disease Prediction',
                           'Parkinsons Prediction'],
                          icons=['activity','heart','person'],
                          default_index=0)
    
    
# Diabetes Prediction Page
if (selected == 'Diabetes Prediction'):
    
    # page title
    st.title('Diabetes Prediction using ML')

    Pregnancies = st.number_input('Number of Pregnancies', min_value=0)
    Glucose = st.number_input('Glucose Level', min_value= 0)
    BloodPressure = st.number_input('Blood Pressure value', min_value=0)
    SkinThickness = st.number_input('Skin Thickness value', min_value=0)
    Insulin = st.number_input('Insulin Level', min_value=0)
    BMI = st.number_input('BMI value', format= "%.3f")
    DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value', format= "%.3f")
    Age = st.number_input('Age of the Person', min_value= 1)

    # code for Prediction
    diab_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        x= [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]
        x= dia_scaler.transform(x)
        diab_prediction = dia_model.predict(x)
        
        if (diab_prediction[0] == 1):
          diab_diagnosis = 'The person is diabetic'
        else:
          diab_diagnosis = 'The person is not diabetic'
        
    st.success(diab_diagnosis)



# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):
    
    # page title
    st.title('Heart Disease Prediction using ML')
    age = st.number_input('Age')
    sex = st.number_input('Sex')
    cp = st.number_input('Chest Pain types')
    trestbps = st.number_input('Resting Blood Pressure')
    chol = st.number_input('Serum Cholestoral in mg/dl')
    fbs = st.number_input('Fasting Blood Sugar > 120 mg/dl ?')
    restecg = st.number_input('Resting Electrocardiographic results')
    thalach = st.number_input('Maximum Heart Rate achieved')
    exang = st.number_input('Exercise Induced Angina')
    oldpeak = st.number_input('ST depression induced by exercise', format="%0.2f")
    slope = st.number_input('Slope of the peak exercise ST segment')
    ca = st.number_input('Major vessels colored by flourosopy')
    thal = st.number_input('thal')
        
    # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]])                          
        
        if (heart_prediction[0] == 1):
          heart_diagnosis = 'The person is having heart disease'
        else:
          heart_diagnosis = 'The person does not have any heart disease'
        
    st.success(heart_diagnosis)
        
    
    
# Parkinson's Prediction Page
if (selected == "Parkinsons Prediction"):
    
    # page title
    st.title("Parkinson's Disease Prediction using ML")
    
    fo = st.number_input('MDVP-Fo(Hz)', format= "%.6f")
    fhi = st.number_input('MDVP-Fhi(Hz)', format= "%.6f")
    flo = st.number_input('MDVP-Flo(Hz)', format= "%.6f")
    Jitter_percent = st.number_input('MDVP-Jitter(%)', format= "%.6f")
    Jitter_Abs = st.number_input('MDVP-Jitter(Abs)', format= "%.6f")
    RAP = st.number_input('MDVP-RAP', format= "%.6f")
    PPQ = st.number_input('MDVP-PPQ', format= "%.6f")
    DDP = st.number_input('Jitter-DDP', format= "%.6f")
    Shimmer = st.number_input('MDVP-Shimmer', format= "%.6f")
    Shimmer_dB = st.number_input('MDVP-Shimmer(dB)', format= "%.6f")
    APQ3 = st.number_input('Shimmer-APQ3', format= "%.6f")
    APQ5 = st.number_input('Shimmer-APQ5', format= "%.6f")
    APQ = st.number_input('MDVP-APQ', format= "%.6f")
    DDA = st.number_input('Shimmer-DDA', format= "%.6f")
    NHR = st.number_input('NHR', format= "%.6f")
    HNR = st.number_input('HNR', format= "%.6f")
    RPDE = st.number_input('RPDE', format= "%.6f")
    DFA = st.number_input('DFA', format= "%.6f")
    spread1 = st.number_input('spread1', format= "%.6f")
    spread2 = st.number_input('spread2', format= "%.6f")
    D2 = st.number_input('D2', format= "%.6f")
    PPE = st.number_input('PPE', format= "%.6f")
        
    # code for Prediction
    parkinsons_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        x= [[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]]
        x= par_scaler.tranform(x)
        
        parkinsons_prediction = par_model.predict(x)                          
        
        if (parkinsons_prediction[0] == 1):
          parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
          parkinsons_diagnosis = "The person does not have Parkinson's disease"
        
    st.success(parkinsons_diagnosis)