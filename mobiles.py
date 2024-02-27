    import os
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import StandardScaler
    import mlflow
    import mlflow.sklearn

    # Specify the root path where your data folders are located
    root_path = "D:/resumeai/data"

    # Read training data
    train_data = pd.read_csv(os.path.join(root_path, "train.csv"))

    # Check if 'id' is present in the columns before dropping
    if 'id' in train_data.columns:
        train_data = train_data.drop('id', axis=1)

    # Feature engineering: Create a new feature 'screen_size'
    train_data['screen_size'] = train_data['sc_h'] * train_data['sc_w'] * train_data['px_height'] * train_data['px_width']

    # Separate features (X) and target variable (y) in training data
    X_train = train_data.drop(['price_range'], axis=1)
    y_train = train_data['price_range']

    # Read testing data
    test_data = pd.read_csv(os.path.join(root_path, "test.csv"))

    # Drop 'id' column from testing data
    if 'id' in test_data.columns:
        test_data = test_data.drop('id', axis=1)

    # Feature engineering: Create a new feature 'screen_size'
    test_data['screen_size'] = test_data['sc_h'] * test_data['sc_w'] * test_data['px_height'] * test_data['px_width']

    # Separate features (X) in testing data
    X_test = test_data

    # Standardize the features (optional, but can be beneficial for some models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Set the experiment name
    experiment_name = "mobile_classification_prediction"
    mlflow.set_experiment(experiment_name)

    # Split the training data for validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

    # Start MLflow run
    with mlflow.start_run():
        # Build a Random Forest Classifier model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train_split, y_train_split)

        # Make predictions on the validation set
        predictions = model.predict(X_val_split)

        # ... (previous code remains unchanged)

        # Evaluate the classification model on the validation set
        accuracy = accuracy_score(y_val_split, predictions)
        classification_report_dict = classification_report(y_val_split, predictions, output_dict=True)

        # Print training and evaluation information
        print("Model trained successfully.")
        print(f'Accuracy on validation set: {accuracy}')
        print('Classification Report:')
        print(classification_report)

        # Log model parameters and metrics to MLflowa
        mlflow.log_param("model", "RandomForestClassifier")
        mlflow.log_param("features", list(X_train.columns))
        mlflow.log_metric("accuracy", accuracy)

        # Log individual metrics from the classification report
        for class_label, metrics in classification_report_dict.items():
            if class_label in ['0', '1', '2', '3']:  # Assuming your class labels are strings
                for metric_name, metric_value in metrics.items():
                    metric_key = f"{class_label}_{metric_name}"
                    mlflow.log_metric(metric_key, metric_value)

        # Log the model using mlflow.sklearn.log_model
        mlflow.sklearn.log_model(model, "model")

        print("Model logged to MLflow.")

        # Now, you can use the trained model to predict the class of a new mobile phone
        new_mobile_features = X_test.iloc[0:1]  # Example features for a new mobile phone
        new_mobile_features_scaled = scaler.transform(new_mobile_features)
        predicted_class = model.predict(new_mobile_features_scaled)

        print(f'Predicted Class for the new mobile phone: {predicted_class[0]}')
