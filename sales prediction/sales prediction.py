# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# Step 1: Load and explore the data
def load_and_explore_data():
    # Load dataset (replace with your actual data path)
    try:
        data = pd.read_csv('advertising.csv')  # Columns: TV, Radio, Newspaper, Sales
        print("Dataset loaded successfully!")
    except FileNotFoundError:
        print("File not found. Using sample data instead.")
        # Create sample data if file not found
        np.random.seed(42)
        data = pd.DataFrame({
            'TV': np.random.normal(150, 50, 200),
            'Radio': np.random.normal(30, 15, 200),
            'Newspaper': np.random.normal(20, 10, 200),
            'Sales': np.random.normal(15, 5, 200)
        })
    
    # Display dataset information
    print("\nDataset Overview:")
    print(data.head())
    
    print("\nDataset Information:")
    print(data.info())
    
    print("\nStatistical Summary:")
    print(data.describe())
    
    return data

# Step 2: Visualize the data
def visualize_data(data):
    print("\nVisualizing relationships between features and sales...")
    
    # Pairplot
    plt.figure(figsize=(12, 6))
    sns.pairplot(data, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', 
                 height=4, aspect=1, kind='scatter')
    plt.suptitle('Relationship Between Advertising Channels and Sales', y=1.02)
    plt.show()
    
    # Correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Advertising Channels and Sales')
    plt.show()

# Step 3: Prepare data for modeling
def prepare_data(data):
    print("\nPreparing data for modeling...")
    
    # Separate features and target
    X = data.drop('Sales', axis=1)
    y = data['Sales']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler

# Step 4: Build and evaluate models
def build_and_evaluate_models(X_train, X_test, y_train, y_test):
    print("\nBuilding and evaluating models...")
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'mse': mse,
            'r2': r2
        }
        
        print(f"\n{name} Performance:")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R-squared Score: {r2:.2f}")
    
    return results

# Step 5: Analyze feature importance
def analyze_feature_importance(model, feature_names):
    print("\nAnalyzing feature importance...")
    
    if hasattr(model, 'feature_importances_'):
        # For tree-based models
        importances = model.feature_importances_
    else:
        # For linear models
        importances = np.abs(model.coef_)
    
    feature_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importances:")
    print(feature_imp)
    
    # Plot feature importances
    plt.figure(figsize=(10, 5))
    sns.barplot(x='Importance', y='Feature', data=feature_imp)
    plt.title('Feature Importances for Sales Prediction')
    plt.show()
    
    return feature_imp

# Step 6: Make predictions on new data
def predict_new_data(model, scaler=None):
    print("\nMaking predictions on new data...")
    
    # Example new advertising budgets
    new_data = pd.DataFrame({
        'TV': [230.1, 44.5, 180.8],
        'Radio': [37.8, 39.3, 10.8],
        'Newspaper': [69.2, 45.1, 58.4]
    })
    
    print("\nNew Advertising Budgets:")
    print(new_data)
    
    # Scale the data if scaler is provided
    if scaler:
        new_data_scaled = scaler.transform(new_data)
        predictions = model.predict(new_data_scaled)
    else:
        predictions = model.predict(new_data)
    
    # Add predictions to the dataframe
    new_data['Predicted_Sales'] = predictions
    
    print("\nPredicted Sales:")
    print(new_data)
    
    return new_data

# Step 7: Save the model
def save_model(model, filename='sales_prediction_model.pkl'):
    joblib.dump(model, filename)
    print(f"\nModel saved successfully as {filename}")

# Main function to run the complete pipeline
def main():
    # Step 1: Load and explore data
    data = load_and_explore_data()
    
    # Step 2: Visualize data
    visualize_data(data)
    
    # Step 3: Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data(data)
    
    # Step 4: Build and evaluate models
    results = build_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Get the best model based on R-squared score
    best_model_name = max(results, key=lambda x: results[x]['r2'])
    best_model = results[best_model_name]['model']
    
    print(f"\nBest model is: {best_model_name} with R-squared: {results[best_model_name]['r2']:.2f}")
    
    # Step 5: Analyze feature importance
    feature_importance = analyze_feature_importance(best_model, X_train.columns)
    
    # Step 6: Make predictions on new data
    # Check if the best model needs scaled data
    if best_model_name == 'Linear Regression':
        new_predictions = predict_new_data(best_model, scaler)
    else:
        new_predictions = predict_new_data(best_model)
    
    # Step 7: Save the best model
    save_model(best_model)
    
    print("\nSales prediction pipeline completed successfully!")

if __name__ == "__main__":
    main()