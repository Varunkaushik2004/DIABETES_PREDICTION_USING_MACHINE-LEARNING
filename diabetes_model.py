import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Load the dataset
file_path = 'diabetes.csv'  # Replace with the path to your diabetes.csv file
data = pd.read_csv(file_path)

# Replace zeros in selected columns with NaN and fill with the median
columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[columns_with_zeros] = data[columns_with_zeros].replace(0, pd.NA)
data.fillna(data.median(), inplace=True)

# Split the data into features and target
X = data.drop(columns='Outcome')
y = data['Outcome']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
logistic_model = LogisticRegression(max_iter=1000, random_state=42)
logistic_model.fit(X_train, y_train)

# Train a Random Forest model
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)

# Save the Logistic Regression model
with open('logistic_model.pkl', 'wb') as log_file:
    pickle.dump(logistic_model, log_file)

# Save the Random Forest model
with open('random_forest_model.pkl', 'wb') as rf_file:
    pickle.dump(random_forest_model, rf_file)

print("Both models have been saved: 'logistic_model.pkl' and 'random_forest_model.pkl'")
