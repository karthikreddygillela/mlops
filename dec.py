import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, UndefinedMetricWarning
import mlflow
import mlflow.sklearn
from tqdm import tqdm
import warnings

# Set the MLflow tracking URI via environment variable
os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"

# Start an MLflow run
mlflow.start_run()

df = pd.read_excel('D:/resumeai/devops.xlsx', header=None)

# Assuming the first column contains resume text and the second column contains labels
X = df[0]  # Extract resume text

y = ['Label'] * len(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer for text data
tfidf_vectorizer = TfidfVectorizer()

# Transform the training and testing text data into numerical TF-IDF features
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Create a list of DevOps-related keywords
keywords = ['ansible']  # Your list of keywords

# Create a CountVectorizer for binary keyword presence features
count_vectorizer = CountVectorizer(vocabulary=keywords, binary=True)

# Transform the training and testing data into binary keyword presence features
X_train_keywords = count_vectorizer.transform(X_train)
X_test_keywords = count_vectorizer.transform(X_test)

# Combine TF-IDF features and keyword presence features
import scipy.sparse as sp
X_train_combined = sp.hstack((X_train_tfidf, X_train_keywords))
X_test_combined = sp.hstack((X_test_tfidf, X_test_keywords))

# Hyperparameters for Multinomial Naive Bayes
alpha = 0.5  # Adjust this value based on your cross-validation results

# Create a Multinomial Naive Bayes classifier with hyperparameters
classifier = MultinomialNB(alpha=alpha)

# Train the classifier on the combined features
classifier.fit(X_train_combined, y_train)

# Make predictions on the testing data
y_pred = classifier.predict(X_test_combined)

# Calculate additional metrics with zero_division parameter
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UndefinedMetricWarning)
    precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)

accuracy = accuracy_score(y_test, y_pred)

# Log metrics to MLflow
mlflow.log_param("keywords", keywords)
mlflow.log_param("alpha", alpha)
mlflow.log_metric("accuracy", accuracy)
mlflow.log_metric("precision", precision)
mlflow.log_metric("recall", recall)
mlflow.log_metric("f1_score", f1)

# Live progress bar with tqdm
epochs = 1000
for epoch in tqdm(range(epochs), desc="Training"):
    # ... Your training steps ...

    # Log metrics to MLflow inside the loop
    mlflow.log_metric("accuracy", accuracy)

    # Log the model periodically or at the end of training
    if epoch % 100 == 0:
        mlflow.sklearn.log_model(classifier, f"devops_epoch_{epoch}")

# Log the final model using MLflow's log_model
mlflow.sklearn.log_model(classifier, "devops_final")

# Get the run ID dynamically
run_id = mlflow.active_run().info.run_id

# Register the model with the actual run ID
mlflow.register_model(f"runs:/{run_id}/devops", "resumeai")

# End the MLflow run
mlflow.end_run()
