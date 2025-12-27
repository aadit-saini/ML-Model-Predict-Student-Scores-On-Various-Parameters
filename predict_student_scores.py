import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
print("Loading dataset...")
data = pd.read_csv('Exam_Score_Prediction.csv')
print("\nFirst 5 rows of the dataset:")
print(data.head())

# Check for missing values
print("\nChecking for missing values:")
print(data.isnull().sum())

# Handle missing values (drop rows with missing data)
data = data.dropna()
print(f"\nDataset shape after removing missing values: {data.shape}")

# Feature selection
print("\nSelecting features...")
X = data[['age', 'gender', 'course', 'study_hours', 'class_attendance',
          'internet_access', 'sleep_hours', 'sleep_quality',
          'study_method', 'facility_rating', 'exam_difficulty']]
y = data['exam_score']

# Encode categorical variables using one-hot encoding
print("Encoding categorical variables...")
X = pd.get_dummies(X, drop_first=True)
print(f"Features after encoding: {X.shape[1]} features")

# Split the dataset into training and testing sets
print("\nSplitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)
print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# Initialize and train the Random Forest Regressor
print("\nTraining Random Forest model...")
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
print("Model training complete!")

# Make predictions
print("\nMaking predictions...")
y_pred = model.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
print("\n" + "="*50)
print(f"Model Performance:")
print(f"R-squared Score: {r2:.4f}")
print("="*50)

# Optional: Display feature importance
print("\nTop 10 Most Important Features:")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False).head(10)
print(feature_importance.to_string(index=False))

# Optional: Plot predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Exam Scores')
plt.ylabel('Predicted Exam Scores')
plt.title('Actual vs Predicted Exam Scores')
plt.tight_layout()
plt.savefig('predictions_plot.png')
print("\nPrediction plot saved as 'predictions_plot.png'")
plt.show()