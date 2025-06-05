import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load Dataset
file_name = r'C:\Users\MOWRIYA\OneDrive\Desktop\IRIS FLOWER CLASSIFICATION\IRIS.csv'

# Check if the file exists
if not os.path.exists(file_name):
    raise FileNotFoundError(f"The file '{file_name}' was not found. Please check the file path.")

# Load the dataset
df = pd.read_csv(file_name)

# Step 2: Validate Dataset
print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nDataset Summary:")
print(df.describe())

# Check for missing values
print("\nNull values:")
print(df.isnull().sum())
if df.isnull().sum().any():
    print("Null values detected. Filling missing values...")
    df = df.fillna(df.mean())

print("\nClass distribution:")
print(df['species'].value_counts())

# Step 3: Visualize Data
# Pairplot
sns.pairplot(df, hue='species')
plt.suptitle("Iris Flower Pairplot", y=1.02)
plt.show()

# Correlation Heatmap
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
if len(numeric_columns) > 0:
    sns.heatmap(df[numeric_columns].corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()
else:
    print("No numeric columns found for correlation heatmap.")

# Step 4: Data Preprocessing
# Encode the species column into numeric values
label_encoder = LabelEncoder()
df['species_encoded'] = label_encoder.fit_transform(df['species'])
print("\nEncoded Species Mapping:")
print(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# Separate features and target
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species_encoded']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTraining and testing data split completed.")

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("\nFeature standardization completed.")

# Step 5: Build and Evaluate Logistic Regression Model
# Train the model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
print("\nLogistic Regression model training completed.")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))