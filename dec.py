import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn

# Start an MLflow run
mlflow.start_run()

df = pd.read_excel('D:/resumeai/devops1.xlsx', header=None)

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
keywords = ['idiot']  # Your list of keywords

# Create a CountVectorizer for binary keyword presence features
count_vectorizer = CountVectorizer(vocabulary=keywords, binary=True)

# Transform the training and testing data into binary keyword presence features
X_train_keywords = count_vectorizer.transform(X_train)
X_test_keywords = count_vectorizer.transform(X_test)

# Combine TF-IDF features and keyword presence features
import scipy.sparse as sp
X_train_combined = sp.hstack((X_train_tfidf, X_train_keywords))
X_test_combined = sp.hstack((X_test_tfidf, X_test_keywords))

# Create a Multinomial Naive Bayes classifier
classifier = MultinomialNB()

# Train the classifier on the combined features
classifier.fit(X_train_combined, y_train)

# Make predictions on the testing data
y_pred = classifier.predict(X_test_combined)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Log metrics to MLflow
mlflow.log_param("keywords", keywords)
mlflow.log_metric("accuracy", accuracy)
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Save the trained model using MLflow
mlflow.sklearn.log_model(classifier, "devops_classifier")

# End the MLflow run
mlflow.end_run()
