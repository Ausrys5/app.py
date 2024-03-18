import streamlit as st
import pandas as pd
import os
from googleapiclient.discovery import build
from google.api_core.client_options import ClientOptions

# Setup environment credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/ausrabruzaite/Desktop/loan-model-417515-c29a3a833426.json"
PROJECT = "loan-model-417515"
REGION = "europe-west2"
MODEL_NAME = "bruzaite_model"

loan_titles = [
    "car", "credit_card", "debt_consolidation", "educational",
    "home_improvement", "house", "major_purchase", "medical",
    "moving", "other", "renewable_energy", "small_business",
    "vacation", "wedding"
]

states = [
    "PA", "NJ", "NY", "MA", "CT", "RI", "NH", "VT", "ME", "DE", "MD", "VA", "WV", "NC", "SC", "GA",
    "FL", "KY", "TN", "MS", "AL", "OK", "TX", "AR", "LA", "IL", "IN", "IA", "KS", "MI", "MN", "MO",
    "NE", "ND", "OH", "SD", "WI", "CA", "AZ", "CO", "ID", "MT", "NV", "NM", "OR", "UT", "WA", "WY",
    "AK", "HI", "DC"
]

st.title('Loan Prediction Application')

# Input fields
amount_requested = st.number_input('Amount Requested', min_value=0)
debt_to_income_ratio = st.number_input('Debt-To-Income Ratio', min_value=0.0, format="%.2f")
employment_length = st.selectbox('Employment Length', options=["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"])
credit_score_category = st.selectbox('Credit Score Category', options=["Poor", "Fair", "Good", "Very Good", "Excellent"])
loan_title = st.selectbox('Loan Title', options=loan_titles)
state = st.selectbox('State', options=states)

# Function to prepare input data
def prepare_input_data(amount_requested, debt_to_income_ratio, employment_length, credit_score_category, loan_title, state):
    # Create a DataFrame for the input features
    input_features = pd.DataFrame({
        'Amount Requested': [amount_requested],
        'Debt-To-Income Ratio': [debt_to_income_ratio],
        'Employment Length': [employment_length],
        'Credit Score Category': [credit_score_category],
        'Loan Title': [loan_title],
        'State': [state]
    })
    
    # Convert ordinal features using the ordering index
    emp_ordering = ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"]
    credit_score_ordering = ["Poor", "Fair", "Good", "Very Good", "Excellent"]
    
    input_features['Employment Length'] = input_features['Employment Length'].apply(lambda x: emp_ordering.index(x) if x in emp_ordering else -1)
    input_features['Credit Score Category'] = input_features['Credit Score Category'].apply(lambda x: credit_score_ordering.index(x) if x in credit_score_ordering else -1)
    
    # One-hot encode nominal features
    input_features = pd.get_dummies(input_features)
    
    # Create a list of all possible columns based on training
    all_possible_columns = [
        'Amount Requested', 'Debt-To-Income Ratio', 'Employment Length', 'Credit Score Category'
    ] + [f'State_{st}' for st in states] + [f'Loan Title_{title}' for title in loan_titles]
    
    # Add missing columns with 0 values
    for column in all_possible_columns:
        if column not in input_features.columns:
            input_features[column] = 0
    
    # Ensure the columns are in the same order as during model training
    input_features = input_features[all_possible_columns]
    
    # Convert the DataFrame to a json-serializable list of lists
    # Each sub-list corresponds to a single prediction instance
    prepared_data = input_features.values.tolist()
    
    return prepared_data

# Function to send json data to a deployed model for prediction
def predict_json(project, region, model, instances, version=None):
    client_options = ClientOptions(api_endpoint=f"https://{region}-ml.googleapis.com")
    service = build('ml', 'v1', client_options=client_options)
    name = f'projects/{project}/models/{model}'
    if version is not None:
        name += f'/versions/{version}'
    
    response = service.projects().predict(name=name, body={'instances': instances}).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])
        print(f"Error making prediction: {error_message}")
        raise RuntimeError(error_message)
    return response['predictions']

# Function to make prediction using the Google Cloud ML Engine
def predict(input_data):
    predictions = predict_json(PROJECT, REGION, MODEL_NAME, input_data)
    print(predictions)
    # Return the prediction result
    return predictions

# Button for prediction
if st.button('Predict Loan Approval'):
    input_data = prepare_input_data(amount_requested, debt_to_income_ratio, employment_length, credit_score_category, loan_title, state)
    predictions = predict(input_data)
    prediction = predictions[0]

    # Display the prediction result
    result = "Approved" if prediction == 1 else "Not Approved"
    st.success(f'Prediction: {result}')