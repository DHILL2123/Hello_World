# Imports
# standard imports
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def decision_tree_model(df, readme):
    # Prepare the features and labels
    X = df[readme]
    y = df['Sentiment']
    # Split the data into train, validate, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    # Vectorize the text data
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)
    # Train the decision tree classifier
    dt_classifier = DecisionTreeClassifier(random_state=42)
    dt_classifier.fit(X_train_vec, y_train)
    # Make predictions on the validation set
    y_val_pred = dt_classifier.predict(X_val_vec)
    # Evaluate the model on validation set
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print('Validation Accuracy:', val_accuracy)
    # Make predictions on the test set
    y_test_pred = dt_classifier.predict(X_test_vec)
    # Evaluate the model on test set
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print('Test Accuracy:', test_accuracy)
    return val_accuracy, test_accuracy



def random_forest_model(df, readme):
    # Prepare the features and labels
    X = df[readme]
    y = df['language']
    # Split the data into train, validate, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    # Vectorize the text data
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)
    # Train the random forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train_vec, y_train)
    # Make predictions on the validation set
    y_val_pred = rf_classifier.predict(X_val_vec)
    # Evaluate the model on validation set
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print('Validation Accuracy:', val_accuracy)
    # Make predictions on the test set
    y_test_pred = rf_classifier.predict(X_test_vec)
    # Evaluate the model on test set
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print('Test Accuracy:', test_accuracy)
    return val_accuracy, test_accuracy



def knn_model(df, readme):
    # Prepare the features and labels
    X = df[readme]
    y = df['language']
    # Split the data into train, validate, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    # Vectorize the text data
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)
    # Train the KNN classifier
    k = 5  # number of neighbors
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train_vec, y_train)
    # Make predictions on the validation set
    y_val_pred = knn_classifier.predict(X_val_vec)
    # Evaluate the model on validation set
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print('Validation Accuracy:', val_accuracy)
    # Make predictions on the test set
    y_test_pred = knn_classifier.predict(X_test_vec)
    # Evaluate the model on test set
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print('Test Accuracy:', test_accuracy)
    return val_accuracy, test_accuracy



def log_reg_model(df, readme):
    # Prepare the features and labels
    X = df[readme]
    y = df['language']
    # Split the data into train, validate, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    # Vectorize the text data
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)
    # Train the logistic regression model
    logreg_model = LogisticRegression()
    logreg_model.fit(X_train_vec, y_train)
    # Make predictions on the validation set
    y_val_pred = logreg_model.predict(X_val_vec)
    # Evaluate the model on the validation set
    val_accuracy = accuracy_score(y_val, y_val_pred)
    # Make predictions on the test set
    y_test_pred = logreg_model.predict(X_test_vec)
    # Evaluate the model on the test set
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print('Train Accuracy:', test_accuracy)
    print('Validation Accuracy:', val_accuracy)
    return val_accuracy, test_accuracy



def create_models_df(readme, model, train_accuracy, validate_accuracy):
    metric_df = pd.DataFrame(data=[
        {'Readme': readme,
         'Model': model,
         'train_accuracy': train_accuracy,
         'validate_accuracy': validate_accuracy,
    }]
    )
    return metric_df



def add_to_models_df(model, readme, metric_df, train_accuracy, validate_accuracy):
    metric_df = metric_df.append(
        {'Readme': readme,
         'Model': model,
         'train_accuracy': train_accuracy,
         'validate_accuracy': validate_accuracy}, ignore_index=True)
    return metric_df




