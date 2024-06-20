{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "536bc5a8",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import joblib\n",
    "import sqlite3\n",
    "from sklearn.ensemble import RandomForestClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "1107a4b2",
   "metadata": {},
   "outputs": [],
   "source": [
    "model=joblib.load('random_forest_model.joblib')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "cf428290",
   "metadata": {},
   "outputs": [],
   "source": [
    "def save_to_db(df):\n",
    "    conn=sqlite3.connect('Heart_Disease_Patients.db')\n",
    "    df.to_sql('Heart_Disease_Patients',conn,if_exists='append',index=False)\n",
    "    conn.close()\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "3c338b39",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-06-20 10:52:52.049 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\skhosanal\\AppData\\Local\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    }
   ],
   "source": [
    "def main():\n",
    "    st.title(\"Heart Disease Prediction App\")\n",
    "    st.write(\"Enter the patient details to predict the likelihood of heart disease.\")\n",
    "      # Define user inputs\n",
    "    age = st.number_input(\"Age\", min_value=1, max_value=120, value=55)\n",
    "    sex = st.selectbox(\"Sex\", [0, 1], format_func=lambda x: \"Male\" if x == 1 else \"Female\")\n",
    "    cp = st.selectbox(\"Chest Pain Type\", [0, 1, 2, 3], format_func=lambda x: [\"Typical angina\", \"Atypical angina\", \"Non-anginal pain\", \"Asymptomatic\"][x])\n",
    "    trestbps = st.number_input(\"Resting Blood Pressure (in mm Hg)\", min_value=80, max_value=200, value=120)\n",
    "    chol = st.number_input(\"Serum Cholesterol (in mg/dl)\", min_value=100, max_value=400, value=200)\n",
    "    fbs = st.selectbox(\"Fasting Blood Sugar > 120 mg/dl\", [0, 1], format_func=lambda x: \"True\" if x == 1 else \"False\")\n",
    "    restecg = st.selectbox(\"Resting ECG Results\", [0, 1, 2], format_func=lambda x: [\"Normal\", \"Having ST-T wave abnormality\", \"Showing probable or definite left ventricular hypertrophy\"][x])\n",
    "    thalach = st.number_input(\"Maximum Heart Rate Achieved\", min_value=60, max_value=220, value=150)\n",
    "    exang = st.selectbox(\"Exercise Induced Angina\", [0, 1], format_func=lambda x: \"Yes\" if x == 1 else \"No\")\n",
    "    oldpeak = st.number_input(\"ST Depression Induced by Exercise\", min_value=0.0, max_value=6.0, value=1.0)\n",
    "    slope = st.selectbox(\"Slope of the Peak Exercise ST Segment\", [0, 1, 2], format_func=lambda x: [\"Upsloping\", \"Flat\", \"Downsloping\"][x])\n",
    "    ca = st.selectbox(\"Number of Major Vessels Colored by Fluoroscopy\", [0, 1, 2, 3, 4])\n",
    "    thal = st.selectbox(\"Thalassemia\", [0, 1, 2], format_func=lambda x: [\"Normal\", \"Fixed Defect\", \"Reversible Defect\"][x])\n",
    "    \n",
    "     # Create a DataFrame for the input data\n",
    "    input_data = pd.DataFrame({\n",
    "        'age': [age],\n",
    "        'sex': [sex],\n",
    "        'cp': [cp],\n",
    "        'trestbps': [trestbps],\n",
    "        'chol': [chol],\n",
    "        'fbs': [fbs],\n",
    "        'restecg': [restecg],\n",
    "        'thalach': [thalach],\n",
    "        'exang': [exang],\n",
    "        'oldpeak': [oldpeak],\n",
    "        'slope': [slope],\n",
    "        'ca': [ca],\n",
    "        'thal': [thal]\n",
    "    })\n",
    "    \n",
    "     # Predict button\n",
    "    if st.button(\"Predict\"):\n",
    "        try:\n",
    "            # Save the input data to the SQLite database\n",
    "            save_to_db(input_data)\n",
    "            # Make the prediction\n",
    "            prediction = model.predict(input_data)\n",
    "            result = \"Patient has Heart Disease!\" if prediction[0] == 1 else \"Patient is safe, No Heart Disease.\"\n",
    "            st.success(f\"The model predicts: {result}\")\n",
    "        except Exception as e:\n",
    "            st.error(f\"An error occurred: {e}\")\n",
    "            \n",
    "if __name__ == '__main__':\n",
    "    main()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "e74d53ae",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: streamlit in c:\\users\\skhosanal\\appdata\\local\\anaconda3\\lib\\site-packages (1.33.0)Note: you may need to restart the kernel to use updated packages.\n",
      "\n",
      "Requirement already satisfied: altair<6,>=4.0 in c:\\users\\skhosanal\\appdata\\local\\anaconda3\\lib\\site-packages (from streamlit) (5.3.0)\n",
      "Requirement already satisfied: blinker<2,>=1.0.0 in c:\\users\\skhosanal\\appdata\\local\\anaconda3\\lib\\site-packages (from streamlit) (1.7.0)\n",
      "Requirement already satisfied: cachetools<6,>=4.0 in c:\\users\\skhosanal\\appdata\\local\\anaconda3\\lib\\site-packages (from streamlit) (5.3.3)\n",
      "Requirement already satisfied: click<9,>=7.0 in c:\\users\\skhosanal\\appdata\\local\\anaconda3\\lib\\site-packages (from streamlit) (8.0.4)\n",
      "Requirement already satisfied: numpy<2,>=1.19.3 in c:\\users\\skhosanal\\appdata\\local\\anaconda3\\lib\\site-packages (from streamlit) (1.24.3)\n",
      "Requirement already satisfied: packaging<25,>=16.8 in c:\\users\\skhosanal\\appdata\\local\\anaconda3\\lib\\site-packages (from streamlit) (23.1)\n",
      "Requirement already satisfied: pandas<3,>=1.3.0 in c:\\users\\skhosanal\\appdata\\local\\anaconda3\\lib\\site-packages (from streamlit) (2.0.3)\n",
      "Requirement already satisfied: pillow<11,>=7.1.0 in c:\\users\\skhosanal\\appdata\\local\\anaconda3\\lib\\site-packages (from streamlit) (9.4.0)\n",
      "Requirement already satisfied: protobuf<5,>=3.20 in c:\\users\\skhosanal\\appdata\\local\\anaconda3\\lib\\site-packages (from streamlit) (4.25.3)\n",
      "Requirement already satisfied: pyarrow>=7.0 in c:\\users\\skhosanal\\appdata\\local\\anaconda3\\lib\\site-packages (from streamlit) (11.0.0)\n",
      "Requirement already satisfied: requests<3,>=2.27 in c:\\users\\skhosanal\\appdata\\local\\anaconda3\\lib\\site-packages (from streamlit) (2.31.0)\n",
      "Requirement already satisfied: rich<14,>=10.14.0 in c:\\users\\skhosanal\\appdata\\local\\anaconda3\\lib\\site-packages (from streamlit) (13.7.1)\n",
      "Requirement already satisfied: tenacity<9,>=8.1.0 in c:\\users\\skhosanal\\appdata\\local\\anaconda3\\lib\\site-packages (from streamlit) (8.2.2)\n",
      "Requirement already satisfied: toml<2,>=0.10.1 in c:\\users\\skhosanal\\appdata\\local\\anaconda3\\lib\\site-packages (from streamlit) (0.10.2)\n",
      "Requirement already satisfied: typing-extensions<5,>=4.3.0 in c:\\users\\skhosanal\\appdata\\local\\anaconda3\\lib\\site-packages (from streamlit) (4.7.1)\n",
      "Requirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in c:\\users\\skhosanal\\appdata\\local\\anaconda3\\lib\\site-packages (from streamlit) (3.1.43)\n",
      "Requirement already satisfied: pydeck<1,>=0.8.0b4 in c:\\users\\skhosanal\\appdata\\local\\anaconda3\\lib\\site-packages (from streamlit) (0.8.1b0)\n",
      "Requirement already satisfied: tornado<7,>=6.0.3 in c:\\users\\skhosanal\\appdata\\local\\anaconda3\\lib\\site-packages (from streamlit) (6.3.2)\n",
      "Requirement already satisfied: watchdog>=2.1.5 in c:\\users\\skhosanal\\appdata\\local\\anaconda3\\lib\\site-packages (from streamlit) (2.1.6)\n",
      "Requirement already satisfied: jinja2 in c:\\users\\skhosanal\\appdata\\local\\anaconda3\\lib\\site-packages (from altair<6,>=4.0->streamlit) (3.1.2)\n",
      "Requirement already satisfied: jsonschema>=3.0 in c:\\users\\skhosanal\\appdata\\local\\anaconda3\\lib\\site-packages (from altair<6,>=4.0->streamlit) (4.17.3)\n",
      "Requirement already satisfied: toolz in c:\\users\\skhosanal\\appdata\\local\\anaconda3\\lib\\site-packages (from altair<6,>=4.0->streamlit) (0.12.0)\n",
      "Requirement already satisfied: colorama in c:\\users\\skhosanal\\appdata\\local\\anaconda3\\lib\\site-packages (from click<9,>=7.0->streamlit) (0.4.6)\n",
      "Requirement already satisfied: gitdb<5,>=4.0.1 in c:\\users\\skhosanal\\appdata\\local\\anaconda3\\lib\\site-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.11)\n",
      "Requirement already satisfied: python-dateutil>=2.8.2 in c:\\users\\skhosanal\\appdata\\local\\anaconda3\\lib\\site-packages (from pandas<3,>=1.3.0->streamlit) (2.8.2)\n",
      "Requirement already satisfied: pytz>=2020.1 in c:\\users\\skhosanal\\appdata\\local\\anaconda3\\lib\\site-packages (from pandas<3,>=1.3.0->streamlit) (2023.3.post1)\n",
      "Requirement already satisfied: tzdata>=2022.1 in c:\\users\\skhosanal\\appdata\\local\\anaconda3\\lib\\site-packages (from pandas<3,>=1.3.0->streamlit) (2023.3)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in c:\\users\\skhosanal\\appdata\\local\\anaconda3\\lib\\site-packages (from requests<3,>=2.27->streamlit) (2.0.4)\n",
      "Requirement already satisfied: idna<4,>=2.5 in c:\\users\\skhosanal\\appdata\\local\\anaconda3\\lib\\site-packages (from requests<3,>=2.27->streamlit) (3.4)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in c:\\users\\skhosanal\\appdata\\local\\anaconda3\\lib\\site-packages (from requests<3,>=2.27->streamlit) (1.26.16)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in c:\\users\\skhosanal\\appdata\\local\\anaconda3\\lib\\site-packages (from requests<3,>=2.27->streamlit) (2023.11.17)\n",
      "Requirement already satisfied: markdown-it-py>=2.2.0 in c:\\users\\skhosanal\\appdata\\local\\anaconda3\\lib\\site-packages (from rich<14,>=10.14.0->streamlit) (2.2.0)\n",
      "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in c:\\users\\skhosanal\\appdata\\local\\anaconda3\\lib\\site-packages (from rich<14,>=10.14.0->streamlit) (2.15.1)\n",
      "Requirement already satisfied: smmap<6,>=3.0.1 in c:\\users\\skhosanal\\appdata\\local\\anaconda3\\lib\\site-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit) (5.0.1)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in c:\\users\\skhosanal\\appdata\\local\\anaconda3\\lib\\site-packages (from jinja2->altair<6,>=4.0->streamlit) (2.1.1)\n",
      "Requirement already satisfied: attrs>=17.4.0 in c:\\users\\skhosanal\\appdata\\local\\anaconda3\\lib\\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (22.1.0)\n",
      "Requirement already satisfied: pyrsistent!=0.17.0,!=0.17.1,!=0.17.2,>=0.14.0 in c:\\users\\skhosanal\\appdata\\local\\anaconda3\\lib\\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.18.0)\n",
      "Requirement already satisfied: mdurl~=0.1 in c:\\users\\skhosanal\\appdata\\local\\anaconda3\\lib\\site-packages (from markdown-it-py>=2.2.0->rich<14,>=10.14.0->streamlit) (0.1.0)\n",
      "Requirement already satisfied: six>=1.5 in c:\\users\\skhosanal\\appdata\\local\\anaconda3\\lib\\site-packages (from python-dateutil>=2.8.2->pandas<3,>=1.3.0->streamlit) (1.16.0)\n"
     ]
    }
   ],
   "source": [
    "pip install streamlit\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "4e3e3e59",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
