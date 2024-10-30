# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("gradient_boosting_model.joblib")

# Load the dataset for column references and drop 'Patient ID' and 'Oncotree Code'
data = pd.read_csv("Breast Cancer.csv").drop(['Patient ID', 'Oncotree Code'], axis=1)

# Define the Streamlit app
st.title("Breast Cancer Prediction App")
st.write("Please enter the following information for prediction:")

# Define preprocessing for categorical and numerical features
categorical_columns = [
    'Type of Breast Surgery', 'Cancer Type', 'Cancer Type Detailed', 'Cellularity', 
    'Chemotherapy', 'Pam50 + Claudin-low subtype', 'Cohort', 'ER status measured by IHC', 
    'ER Status', 'HER2 status measured by SNP6', 'HER2 Status', 'Tumor Other Histologic Subtype', 
    'Hormone Therapy', 'Inferred Menopausal State', 'Integrative Cluster', 
    'Primary Tumor Laterality', 'Overall Survival Status', 'PR Status', 'Radio Therapy', 
    'Relapse Free Status', 'Sex', '3-Gene classifier subtype', "Patient's Vital Status"
]

# Gather user input based on feature columns
input_data = {}
for column in data.columns:
    if column in categorical_columns:
        input_data[column] = st.selectbox(f"{column}", options=data[column].unique())
    else:
        input_data[column] = st.number_input(f"{column}", value=float(data[column].mean()))

# Convert user inputs into a DataFrame and one-hot encode it
input_df = pd.DataFrame([input_data])
input_df = pd.get_dummies(input_df)

# Align with model's expected columns, filling missing ones with 0
input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

# Convert the DataFrame to a numpy array
input_array = input_df.to_numpy().astype(np.float64)

# Make a prediction with the model
prediction = model.predict(input_array)[0]

# Display the prediction result
st.write("Prediction for Patient's Vital Status:", prediction)





