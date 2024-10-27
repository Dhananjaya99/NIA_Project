import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import streamlit as st
import joblib
import pickle

# Title of the Streamlit app
st.title("Impact of Mobile Phone on Students' Health")

# Upload the CSV file
uploaded_file = st.file_uploader('Impact_of_Mobile_Phone_on_Students_Health (3).csv', type="csv")
if uploaded_file is not None:
    # Read the uploaded CSV file
    data = pd.read_csv('Impact_of_Mobile_Phone_on_Students_Health (3).csv')
    
    # Display the first few rows of the dataset
    st.write("## First few rows of the dataset")
    st.write(data.head())
    
    # Check for missing values
    st.write("\n## Missing values in each column")
    st.write(data.isnull().sum())
    
    # Handle missing values by dropping rows with missing values
    data = data.dropna()
    
    # Display basic statistics of the dataset
    st.write("\n## Statistical Summary of the dataset")
    st.write(data.describe())
    
    # Display data types
    st.write("\n## Data types")
    st.write(data.dtypes)
    
    # Convert categorical columns using Label Encoding
    labelencoder = LabelEncoder()
    for column in data.select_dtypes(include=['object']).columns:
        data[column] = labelencoder.fit_transform(data[column])
    
    # Exploratory Data Analysis (EDA)
    st.write("## Correlation Matrix")
    plt.figure(figsize=(15, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    st.pyplot(plt.gcf())
    
    # Pairplot to visualize relationships
    st.write("## Pairplot")
    sns.pairplot(data)
    st.pyplot(plt.gcf())
    
    # Feature Selection and Target Variable
    X = data.iloc[:, :-1]  # Features
    y = data.iloc[:, -1]   # Target (health impact)
    
    # Train/Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    st.write("\n## Model Accuracy")
    st.write(accuracy_score(y_test, y_pred))
    
    st.write("\n## Classification Report")
    st.text(classification_report(y_test, y_pred))
    
    # Feature Importance Plot
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    st.write("## Feature Importance")
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
    plt.ylabel('Importance Score')
    plt.xlabel('Features')
    plt.tight_layout()
    st.pyplot(plt.gcf())
    
    # Save the model using joblib
    joblib.dump(model, 'Impact_of_Mobile_Phone_on_Students_Health_Model.joblib')
    
    # Save the model using pickle
    filename = 'Impact_of_Mobile_Phone_on_Students_Health_Model.sav'
    pickle.dump(model, open(filename, 'wb'))
else:
    st.write('Impact_of_Mobile_Phone_on_Students_Health (3).csv')
