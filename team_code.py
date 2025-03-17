#!/usr/bin/env python

# Edit this script to add your team's code.

################################################################################
# Optional libraries and functions. You can change or remove them.
################################################################################

from helper_code import *
import numpy as np, os, sys
import pandas as pd
import io
import mne
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import joblib

################################################################################
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')
        
    patient_ids, data, label, features = load_challenge_data(data_folder)
    num_patients = len(patient_ids)

    if num_patients == 0:
        raise FileNotFoundError('No data is provided.')
        
    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)
    
    # Train the models.
    if verbose >= 1:
        print('Training the Challenge models on the Challenge data...')

    # Feature selection (column dropping)
    selected_variables = list(data.columns)

    # Save the selected features to file.
    with open(os.path.join(model_folder, 'selected_variables.txt'), 'w') as f:
        f.write("\n".join(selected_variables))

    # Preprocessing: dummy encoding     
    data = pd.get_dummies(data)
    dummy_columns = list(data.columns)
    #Saving the dummy-encoded column names for later alignment during inference.
    with open(os.path.join(model_folder, 'dummy_columns.txt'), 'w') as f:
        f.write("\n".join(dummy_columns))
        
        
    # Define parameters for random forest classifier and regressor.
    n_estimators   = 123  # Number of trees in the forest.
    max_leaf_nodes = 456  # Maximum number of leaf nodes in each tree.
    random_state   = 789  # Random state; set for reproducibility.

    # Impute any missing features; use the mean value by default.
    imputer = SimpleImputer().fit(data)

    # Train the model.
    data_imputed = imputer.transform(data)
    prediction_model = RandomForestClassifier(
        n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(data_imputed, label.ravel())

    # Save the trained model.
    save_challenge_model(model_folder, imputer, prediction_model, selected_variables, dummy_columns)

    if verbose >= 1:
        print('Done!')
        
# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_model(model_folder, verbose):
    if verbose >= 1:
        print('Loading the model...')

    # Load the selected features
    try:
        with open(os.path.join(model_folder, 'selected_variables.txt'), 'r') as f:
            selected_features = f.read().splitlines()
        if verbose:
            print("Loaded selected features from 'selected_variables.txt'")
    except Exception as e:
        if verbose:
            print("Warning: Could not load 'selected_variables.txt'. Using all features. Error:", e)
        selected_features = None

    # Load the dummy-encoded columns
    try:
        with open(os.path.join(model_folder, 'dummy_columns.txt'), 'r') as f:
            dummy_columns = f.read().splitlines()
    except Exception as e:
        if verbose:
            print("Warning: Could not load 'dummy_columns.txt'.", e)
        dummy_columns = None

    # Load the saved model.
    model = joblib.load(os.path.join(model_folder, 'model.sav'))
    model['selected_variables'] = selected_features
    model['dummy_columns'] = dummy_columns
    return model

# Run the trained model on test data.
def run_challenge_model(model, df, verbose):
    imputer = model['imputer']
    prediction_model = model['prediction_model']
    dummy_columns = model['dummy_columns']
    selected_variable = model['selected_variables']

    # Preprocess: apply dummy encoding and align with training dummy columns.
    df = pd.get_dummies(df)
    df = df.reindex(columns=dummy_columns, fill_value=0)
    
    # Impute missing data.
    df_imputed = imputer.transform(df)
    
    # Get prediction probabilities.
    prediction_probability = prediction_model.predict_proba(df_imputed)[:, 1]
    
    # Set a probability threshold.
    threshold = 0.08
    
    # Compute binary predictions using the threshold.
    prediction_binary = (prediction_probability >= threshold).astype(int)
    
    # Write the threshold to a file.
    with open("threshold.txt", "w") as f:
        f.write(str(threshold))
    
    if 'studyid_adm' in df.columns:
        patient_ids = df['studyid_adm'].tolist()
    else:
        raise ValueError("Test data must include the 'studyid_adm' column.")

################################################################################
# Optional functions. You can change or remove these functions and/or add new functions.
################################################################################

# Save the trained model.
def save_challenge_model(model_folder, imputer, prediction_model, selected_variables, dummy_columns):
    d = {
        'imputer': imputer,
        'prediction_model': prediction_model,
        'selected_variables': selected_variables,
        'dummy_columns': dummy_columns,
    }
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)
