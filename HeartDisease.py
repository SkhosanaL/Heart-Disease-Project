import streamlit as st
import pandas as pd
import joblib  

#Loading the trained Model
#model_path = (r"C:\Users\skhosanal\OneDrive - Inkomati-Usuthu Catchment Management Agency\Python Scripts\Streamlit\random_forest_model.joblib")
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

st.header('Enter Patient Details:')

age = st.text_input('Enter Your Age:')
sex = st.text_input('Sex (0: Female, 1: Male):')
cp = st.text_input('Chest Pain Type (0: Asymptomatic, 1: Typical Angina, 2: Atypical Angina, 3: Non-Anginal Pain):')
trestbps =st.text_input('Resting Blood Pressure (mm Hg):')
chol = st.text_input('Serum Cholesterol (mg/dl):')
fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl (0: False, 1: True):')
restecg = st.text_input('Resting Electrocardiographic Results (0 = normal; 1 =abnormal; 2= ventricular Hypertrophy:')
thalach = st.text_input('Maximum Heart Rate Achieved:')
exang = st.text_input('Exercise Induced Angina (0: No, 1: Yes):')
oldpeak = st.text_input('ST Depression Induced by Exercise (Relative to Rest):')
slope = st.text_input('Slope of the Peak Exercise ST Segment (0: Upsloping, 1: Flat, 2: Downsloping):')
ca = st.text_input('Number of Major Vessels Colored by Fluoroscopy (0=Normal, 1=mild, 2=moderate, 3=Severe):')
thal = st.text_input('Thalassemia (1: Normal, 2: Fixed Defect, 3: Reversible Defect):')


# Make predictions
if st.button('Predict'):
    try:
        age = int(age)
        sex = int(sex)
        cp = int(cp)
        trestbps = int(trestbps)
        chol = int(chol)
        fbs = int(fbs)
        restecg = int(restecg)
        thalach = int(thalach)
        exang = int(exang)
        oldpeak = float(oldpeak)
        slope = int(slope)
        ca = int(ca)
        thal = int(thal)
        
        prediction = predict(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
        st.subheader('Prediction')
        st.write('Patient has a Heart Disease' if prediction == 1 else 'Patient has No Heart Disease')
    except ValueError:
        st.error('Please enter valid inputs for all fields.')