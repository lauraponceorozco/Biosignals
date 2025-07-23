from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from IPython.display import display
import os
from help_func_new_paradigm import *
import joblib
from sklearn.preprocessing import StandardScaler

# Resolve path to current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'svm_classifier.pkl')

# Load the classifier
clf = joblib.load(model_path)

# Load the recordings from one trial
base_path = Path.cwd().parent  / "Biosignals" / "Application_Real_Time"/ "data_real_time" / "cur_rec"   
method = ["A1"]
feature_types=["B3"]                                                  
df_combined = process_and_combine_all_csvs(base_path,blink_rejection=False, normalization=method)
X, y = extract_features_from_df_combined(df_combined, fs=128, feature_types = feature_types)



#print("X: ", X)
#print("X shape: ", X.shape)
#print("y: ", y)

# === Feature Scaling ===
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_probs = clf.predict_proba(X_scaled)[:, 1]

print("y_probs: ", y_probs)