#!/usr/bin/env python
# coding: utf-8

# In[4]:


#importing necessary libraries
import pandas as pd
import sqlite3
import seaborn as sns
from tkinter import Tk 
import matplotlib.pyplot as plt

#connecting to the csv file to create a dataframe
df= pd.read_csv(r"C:\Users\skhosanal\OneDrive - Inkomati-Usuthu Catchment Management Agency\Python Scripts\Streamlit\heart.csv")

print("Dataframe created")


# In[5]:





# In[6]:





# In[4]:


#Establishing a connection to sqlite database
conn=sqlite3.connect('Heart.db')
print("Connection established")


# In[ ]:


#converting the dataframe into a database in sqlite.
df.to_sql('Heart',conn,if_exists='replace')

print("dataframe converted into a Table Database")


# In[ ]:


df.head(10)


# In[ ]:


df.shape


# In[ ]:


df.isnull().sum()


# In[ ]:


data_dup=df.duplicated().any


# In[ ]:


df.info()


# In[ ]:


data_dup


# In[ ]:


df.describe()


# In[ ]:


df.shape


# In[ ]:


#Fetching the data from sqlite Heart_Disease_Patients Database.
conn=sqlite3.connect('Heart.db')
cur=conn.cursor()
cur.execute("SELECT * FROM Heart")
rows = cur.fetchall()

for row in rows:
    print(row)

conn.commit()
conn.close()


# In[ ]:


df.dtypes


# In[ ]:


df.shape


# In[ ]:





# In[ ]:


#setting up the matplotlib figure
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 15))
axes = axes.flatten()
categorical_vars = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
# Plot countplots for each categorical variable
for i, var in enumerate(categorical_vars):
    sns.countplot(x=var, hue='target', data=df, ax=axes[i])
    axes[i].set_title(f'Distribution of {var} by target')
    axes[i].set_xlabel(var)
    axes[i].set_ylabel('Count')
    
plt.tight_layout()
plt.show()


# In[ ]:


# Plot histograms for numeric variables based on the target variable
numeric_vars = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

for var in numeric_vars:
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x=var, hue='target', kde=True)
    plt.title(f'Distribution of {var} by Target')
    plt.xlabel(var)
    plt.ylabel('Count')
    plt.legend(title='Target', labels=['No Heart Disease', 'Heart Disease'])
    plt.show()


# In[ ]:


#Data Processing
cate_val = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']
cont_val =['age', 'trestbps', 'chol', 'thalach', 'oldpeak']


# In[ ]:


cate_val


# In[ ]:


cont_val


# In[ ]:


df['cp'].unique()


# In[ ]:


df.head()


# In[ ]:


#feature Scaling
from sklearn.preprocessing import StandardScaler


# In[ ]:


st=  StandardScaler()
df[cont_val]= st.fit_transform(df[cont_val])


# In[ ]:


df.head()


# In[ ]:


#Splitting The dataset into Training Set and Test Set
X = df.drop('target',axis=1)# dependent variable
y = df['target']


# In[ ]:


y


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


y_test


# In[ ]:


X_test


# In[ ]:


df.head()


# In[ ]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression


# In[ ]:

#performing LR prediction
log = LogisticRegression()
log.fit(X_train,y_train)
y_pred1 = log.predict(X_test)


# In[ ]:

#importing accuracy_score
from sklearn.metrics import accuracy_score


# In[ ]:

#generating accuracy_score
accuracy_score(y_test,y_pred1)


# In[ ]:

#importing svm
from sklearn import svm


# In[ ]:

#prediction through SVM 
svm = svm.SVC()
svm.fit(X_train,y_train)
y_pred2 = svm.predict(X_test)
accuracy_score(y_test,y_pred2)


# In[ ]:

#import RFC Model
from sklearn.ensemble import RandomForestClassifier


# In[ ]:

#RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred3 = rf.predict(X_test)
accuracy_score(y_test,y_pred3)


# In[ ]:

#Accuracy scores of the different models
final_data = pd.DataFrame({'Models': ['LR','SVM','RFC'],
                           'ACC': [accuracy_score(y_test,y_pred1),
                                  accuracy_score(y_test,y_pred2),
                                  accuracy_score(y_test,y_pred3)]})


# In[ ]:

#ACC_SCORES
final_data


# In[ ]:

#dropping the target variable on the X-axis
X = df.drop('target',axis=1)# dependent variable
y = df['target']


# In[ ]:


X.shape


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:

#RFC
rf=RandomForestClassifier()
rf.fit(X,y)


# In[ ]:

#Testing the RFC model with new user_input
new_data=pd.DataFrame({
    'age':52,
    'sex':1,
    'cp':0,
    'trestbps':125,
    'chol':212,
    'fbs':0,
    'restecg':1,
    'thalach':168,
    'exang':0,
    'oldpeak':1.8,
    'slope': 2,
    'ca': 2,
    'thal':3,
},index=[0])


# In[ ]:

#Viewing the results of the RFC Model
new_data


# In[ ]:

#Performing the actual prediction on the new data
p=rf.predict(new_data)
if p[0]==0:
    print("Patient is safe, No heart disease.")
else:
    print("Patient is Likely to develop a heart disease!")


# In[ ]:





# In[ ]:
#importing Joblib for saving and loading the RFC Model

import joblib


# In[ ]:

#Saving the model
joblib.dump(rf,'random_forest_model.joblib')


# In[ ]:

#loading the model
model=joblib.load('random_forest_model.joblib')


# In[ ]:

#Performing Prediction
model.predict(new_data)


# In[ ]:




