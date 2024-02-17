# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 22:23:27 2024

@author: BISHWAJIT
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from prettytable import PrettyTable


df = pd.read_csv(r'C:\Users\NITR_CS_PL4K\Downloads\JM1.csv')

# Split the dataframe into features (X) and target (y)
X = df.drop(columns=['Defective'])  # Replace 'target_column' with the name of your target column
y = df['Defective']

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the training and testing sets
print("Training set shape (features):", X_train.shape)
print("Testing set shape (features):", X_test.shape)
print("Training set shape (target):", y_train.shape)
print("Testing set shape (target):", y_test.shape)

# Convert DataFrame to NumPy array
X_train_np = X_train.values
X_test_np = X_test.values

# Initialize classifiers with different kernels
classifiers = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "k-Nearest Neighbors": KNeighborsClassifier(),
    "SVM Sigmoid": SVC(kernel='sigmoid', random_state=42)
}


    # Rest of the code remains the same...


# Initialize dictionaries to store evaluation metrics
metrics_dict = {
    "Classifier": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1 Score": [],
}


# Iterate through each classifier
for clf_name, classifier in classifiers.items():
    classifier.fit(X_train_np, y_train)  # Use NumPy arrays instead of DataFrames
    predictions = classifier.predict(X_test_np)
    
    
    # Calculate evaluation metrics
    conf_matrix = confusion_matrix(y_test, predictions)
    
    # Create confusion matrix table
    conf_matrix_table = PrettyTable()
    conf_matrix_table.title = f"Confusion Matrix for {clf_name}"
    conf_matrix_table.field_names = ["", "Predicted 0", "Predicted 1"]
    
    # Add rows for each actual class
    for i in range(conf_matrix.shape[0]):
        conf_matrix_table.add_row([f"Actual {i}", conf_matrix[i][0], conf_matrix[i][1]])
    
    # Print confusion matrix for each classifier
    print(conf_matrix_table)

            
    # Calculate metrics
    accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
    precision = np.where(np.sum(conf_matrix, axis=0) == 0, 0, np.diag(conf_matrix) / np.sum(conf_matrix, axis=0))
    recall = np.where(np.sum(conf_matrix, axis=1) == 0, 0, np.diag(conf_matrix) / np.sum(conf_matrix, axis=1))
    f1_score = np.where((precision + recall) == 0, 0, 2 * precision * recall / (precision + recall))

    # Store evaluation metrics in the dictionary
    metrics_dict["Classifier"].append(clf_name)
    metrics_dict["Accuracy"].append(accuracy)
    metrics_dict["Precision"].append(np.mean(precision))
    metrics_dict["Recall"].append(np.mean(recall))
    metrics_dict["F1 Score"].append(np.mean(f1_score))
    
# Create PrettyTable for evaluation metrics
metrics_table = PrettyTable()
metrics_table.field_names = ["Classifier", "Accuracy", "Precision", "Recall", "F1 Score"]
for i in range(len(metrics_dict["Classifier"])):
    metrics_table.add_row([
        metrics_dict["Classifier"][i],
        metrics_dict["Accuracy"][i],
        metrics_dict["Precision"][i],
        metrics_dict["Recall"][i],
        metrics_dict["F1 Score"][i]
    ])

# Print evaluation metrics table
print("Evaluation Metrics:")
print(metrics_table)