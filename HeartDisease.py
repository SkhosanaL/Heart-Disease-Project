import streamlit as st
import pandas as pd
import joblib  # type: ignore

#Loading the trained Model
model = joblib.load('random_forest_model.joblib')

st.title('Heart Disease Prediction App')

# Display the attribute descriptions for user guidance

st.sidebar.header('Attribute Descriptions')
st.sidebar.markdown("""
- **age**: Patient's age in years (numerical)
- **sex**: Sex (0 = female; 1 = male) (numerical)
- **cp**: Chest pain type (0 = asymptomatic, 1 = typical angina, 2 = atypical angina, 3 = non-anginal pain)
- **trestbps**: Resting blood pressure (in mm Hg on admission to the hospital) (numerical)
- **chol**: Serum cholesterol in mg/dl (numerical)
- **fbs**: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false) (numerical)
- **restecg**: Resting electrocardiographic results (0 = normal; 1 =abnormal; 2= ventricular Hypertrophy)
- **thalach**: Maximum heart rate achieved (numerical)
- **exang**: Exercise induced angina (1 = yes; 0 = no) (numerical)
- **oldpeak**: ST depression induced by exercise relative to rest (numerical)
- **slope**: The slope of the peak exercise ST segment (0 = upsloping; 1 = flat; 2 = downsloping) 
- **ca**: Number of major vessels colored by fluoroscopy (0-4) (numerical)
- **thal**: Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect) (numerical)
""")

# Function to make prediction
def predict(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    user_input = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal]
    })

    prediction = model.predict(user_input)
    return prediction[0]

# Accept user inputs
st.sidebar.header('User Input Parameters')

age = st.sidebar.number_input('Age (years)', min_value=1, max_value=120, value=29)
sex = st.sidebar.selectbox('Sex (0: Female, 1: Male)', (0, 1))
cp = st.sidebar.selectbox('Chest Pain Type (0: Asymptomatic, 1: Typical Angina, 2: Atypical Angina, 3: Non-Anginal Pain)', (0, 1, 2, 3))
trestbps = st.sidebar.number_input('Resting Blood Pressure (mm Hg)', min_value=80, max_value=200, value=120)
chol = st.sidebar.number_input('Serum Cholesterol (mg/dl)', min_value=100, max_value=600, value=200)
fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl (0: False, 1: True)', (0, 1))
restecg = st.sidebar.selectbox('Resting Electrocardiographic Results (0 = normal; 1 =abnormal; 2= ventricular Hypertrophy'), (0, 1, 2)
thalach = st.sidebar.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=220, value=150)
exang = st.sidebar.selectbox('Exercise Induced Angina (0: No, 1: Yes)', (0, 1))
oldpeak = st.sidebar.number_input('ST Depression Induced by Exercise (Relative to Rest)', min_value=0.0, max_value=6.0, value=1.0)
slope = st.sidebar.selectbox('Slope of the Peak Exercise ST Segment (0: Upsloping, 1: Flat, 2: Downsloping)', (0, 1, 2))
ca = st.sidebar.selectbox('Number of Major Vessels Colored by Fluoroscopy (0-3)', (0, 1, 2, 3, 4))
thal = st.sidebar.selectbox('Thalassemia (1: Normal, 2: Fixed Defect, 3: Reversible Defect)', (1, 2, 3))

#Displaying user inputs
st.subheader('User Input Parameters')
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'cp': [cp],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs],
    'restecg': [restecg],
    'thalach': [thalach],
    'exang': [exang],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'ca': [ca],
    'thal': [thal]
})
st.write(input_data)

# Make predictions
if st.button('Predict'):
    prediction = predict(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    st.subheader('Prediction')
    st.write('Patient has a Heart Disease' if prediction == 1 else 'Patient has No Heart Disease')