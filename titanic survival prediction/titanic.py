# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                            confusion_matrix, roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

## Step 1: Load and Explore the Data
def load_data(file_path):
    """Load the Titanic dataset"""
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully!")
    print(f"\nShape of dataset: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nDataset information:")
    print(df.info())
    print("\nSummary statistics:")
    print(df.describe(include='all'))
    print("\nMissing values per column:")
    print(df.isnull().sum())
    return df

# Load the data (update the file path as needed)
titanic_df = load_data('titanic.csv')

## Step 2: Data Preprocessing
def preprocess_data(df):
    """Preprocess the Titanic dataset"""
    # Make a copy of the original dataframe
    df = df.copy()
    
    # Drop columns that won't be useful for prediction
    df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    
    # Create new features
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Create age groups
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 20, 40, 120], 
                           labels=['Child', 'Teen', 'Adult', 'Elderly'])
    
    # Create fare groups
    df['FareGroup'] = pd.qcut(df['Fare'], 4, labels=['Low', 'Medium', 'High', 'Very High'])
    
    # Convert categorical features
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=['AgeGroup', 'FareGroup', 'Pclass'], drop_first=True)
    
    return df

# Preprocess the data
processed_df = preprocess_data(titanic_df)

## Step 3: Handle Missing Values
def handle_missing_values(df):
    """Handle missing values in the dataset"""
    # Impute missing age values with median
    age_imputer = SimpleImputer(strategy='median')
    df['Age'] = age_imputer.fit_transform(df[['Age']])
    
    # Impute missing embarked values with mode
    embarked_imputer = SimpleImputer(strategy='most_frequent')
    df['Embarked'] = embarked_imputer.fit_transform(df[['Embarked']])
    
    # Impute missing fare values with median
    fare_imputer = SimpleImputer(strategy='median')
    df['Fare'] = fare_imputer.fit_transform(df[['Fare']])
    
    return df

# Handle missing values
processed_df = handle_missing_values(processed_df)

## Step 4: Exploratory Data Analysis (EDA)
def perform_eda(df):
    """Perform exploratory data analysis"""
    # Survival rate by gender
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Sex', y='Survived', data=df)
    plt.title('Survival Rate by Gender (0 = Male, 1 = Female)')
    plt.ylabel('Survival Rate')
    plt.show()
    
    # Survival rate by passenger class (original Pclass)
    if 'Pclass' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Pclass', y='Survived', data=df)
        plt.title('Survival Rate by Passenger Class')
        plt.ylabel('Survival Rate')
        plt.show()
    
    # Age distribution of survivors vs non-survivors
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='Age', hue='Survived', kde=True, bins=30, alpha=0.6)
    plt.title('Age Distribution of Survivors vs Non-Survivors')
    plt.show()
    
    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    plt.show()
    
    # Pairplot of selected features
    sns.pairplot(df[['Age', 'Fare', 'Sex', 'Survived', 'FamilySize']], hue='Survived')
    plt.suptitle('Pairplot of Selected Features', y=1.02)
    plt.show()

# Perform EDA
perform_eda(processed_df)

## Step 5: Prepare Data for Modeling
def prepare_data(df):
    """Prepare data for machine learning"""
    # Separate features and target
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale numerical features
    numeric_features = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)],
        remainder='passthrough')
    
    # Apply transformations
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, preprocessor

# Prepare the data
X_train, X_test, y_train, y_test, preprocessor = prepare_data(processed_df)

## Step 6: Build and Train the Model
def train_model(X_train, y_train):
    """Train a Random Forest classifier with hyperparameter tuning"""
    # Define the model
    rf = RandomForestClassifier(random_state=42)
    
    # Define hyperparameters for tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='accuracy',
        verbose=1
    )
    
    print("\nTraining model with hyperparameter tuning...")
    grid_search.fit(X_train, y_train)
    
    print("\nBest parameters found:")
    print(grid_search.best_params_)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    return best_model

# Train the model
model = train_model(X_train, y_train)

## Step 7: Evaluate the Model
def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluate the model's performance"""
    # Make predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, train_preds)
    test_accuracy = accuracy_score(y_test, test_preds)
    roc_auc = roc_auc_score(y_test, test_probs)
    
    print("\nModel Evaluation:")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test ROC AUC: {roc_auc:.4f}")
    
    # Classification report
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, test_preds))
    
    # Confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, test_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Survived', 'Survived'], 
                yticklabels=['Not Survived', 'Survived'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, test_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    
    return test_accuracy

# Evaluate the model
accuracy = evaluate_model(model, X_train, X_test, y_train, y_test)

## Step 8: Feature Importance Analysis
def analyze_feature_importance(model, preprocessor, df):
    """Analyze feature importance"""
    # Get feature names
    numeric_features = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
    other_features = [col for col in df.columns 
                     if col not in numeric_features + ['Survived']]
    feature_names = numeric_features + other_features
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create a DataFrame for visualization
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()
    
    return feature_importance

# Analyze feature importance
feature_importance = analyze_feature_importance(model, preprocessor, processed_df)

## Step 9: Save the Model
def save_model(model, preprocessor, file_path='titanic_model.joblib'):
    """Save the trained model and preprocessor"""
    # Create a dictionary with model and preprocessor
    model_package = {
        'model': model,
        'preprocessor': preprocessor
    }
    
    # Save to file
    joblib.dump(model_package, file_path)
    print(f"\nModel saved successfully to {file_path}")

# Save the model
save_model(model, preprocessor)

## Step 10: Create a Prediction Function
def predict_survival(model_package, passenger_data):
    """
    Predict survival for new passenger data
    passenger_data should be a dictionary with the following keys:
    {
        'Sex': 'male' or 'female',
        'Age': numeric value,
        'Pclass': 1, 2, or 3,
        'SibSp': integer,
        'Parch': integer,
        'Fare': numeric value,
        'Embarked': 'S', 'C', or 'Q'
    }
    """
    # Convert passenger data to DataFrame
    passenger_df = pd.DataFrame([passenger_data])
    
    # Preprocess the data (same steps as training)
    passenger_df['FamilySize'] = passenger_df['SibSp'] + passenger_df['Parch'] + 1
    passenger_df['IsAlone'] = (passenger_df['FamilySize'] == 1).astype(int)
    passenger_df['AgeGroup'] = pd.cut(passenger_df['Age'], 
                                    bins=[0, 12, 20, 40, 120], 
                                    labels=['Child', 'Teen', 'Adult', 'Elderly'])
    passenger_df['FareGroup'] = pd.qcut(passenger_df['Fare'], 4, 
                                      labels=['Low', 'Medium', 'High', 'Very High'])
    passenger_df['Sex'] = passenger_df['Sex'].map({'male': 0, 'female': 1})
    passenger_df['Embarked'] = passenger_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    passenger_df = pd.get_dummies(passenger_df, 
                                 columns=['AgeGroup', 'FareGroup', 'Pclass'], 
                                 drop_first=True)
    
    # Ensure all expected columns are present
    expected_columns = model_package['preprocessor'].transformers_[0][2] + \
                     [col for col in model_package['preprocessor'].named_transformers_['remainder'].get_feature_names_out()
                      if col != 'Survived']
    
    for col in expected_columns:
        if col not in passenger_df.columns:
            passenger_df[col] = 0
    
    # Reorder columns to match training data
    passenger_df = passenger_df[expected_columns]
    
    # Apply preprocessing
    passenger_processed = model_package['preprocessor'].transform(passenger_df)
    
    # Make prediction
    prediction = model_package['model'].predict(passenger_processed)
    probability = model_package['model'].predict_proba(passenger_processed)[:, 1]
    
    return {
        'prediction': 'Survived' if prediction[0] == 1 else 'Did not survive',
        'probability': float(probability[0])
    }

# Example usage (after loading the model)
# loaded_model = joblib.load('titanic_model.joblib')
# passenger_example = {
#     'Sex': 'female',
#     'Age': 25,
#     'Pclass': 1,
#     'SibSp': 0,
#     'Parch': 0,
#     'Fare': 50,
#     'Embarked': 'C'
# }
# result = predict_survival(loaded_model, passenger_example)
# print(result)

print("\nTitanic Survival Prediction Project Completed Successfully!")