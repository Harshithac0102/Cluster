# # Setup and Configuration

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib

# Check versions to ensure compatibility
import sklearn
print(f"scikit-learn version: {sklearn.__version__}")

import json
import zipfile
import shutil

# Kaggle API credentials
kaggle_credentials = {
    "username": "harshithac0102",
    "key": "d60fb0e6f77a8fd113bfadb0aac4cb09"
}

# Write to kaggle.json
with open("kaggle.json", "w") as file:
    json.dump(kaggle_credentials, file)

print("kaggle.json file created successfully!")

def download_kaggle_dataset(dataset_path):
    # Ensure Kaggle API is installed
    os.system('pip install -q kaggle')

    # Check for kaggle.json file
    if not os.path.isfile('kaggle.json'):
        print("kaggle.json file not found. Please upload it.")
        return

    # Set up Kaggle directory (~/.kaggle on Unix, C:/Users/<user>/.kaggle on Windows)
    kaggle_dir = os.path.join(os.path.expanduser('~'), '.kaggle')
    if not os.path.exists(kaggle_dir):
        os.makedirs(kaggle_dir)

    # Copy kaggle.json to the .kaggle directory
    shutil.copy('kaggle.json', os.path.join(kaggle_dir, 'kaggle.json'))
    print(f"kaggle.json copied to {kaggle_dir}")

    # Download the dataset
    os.system(f'kaggle datasets download -d {dataset_path}')

    # Unzip the dataset
    zip_file = f'{dataset_path.split("/")[-1]}.zip'
    if os.path.exists(zip_file):
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall('.')  # Extract in current directory
        print(f"{zip_file} extracted successfully!")
    else:
        print(f"Failed to find {zip_file}. Download may have failed.")

# Usage:
download_kaggle_dataset('bittlingmayer/amazonreviews')
