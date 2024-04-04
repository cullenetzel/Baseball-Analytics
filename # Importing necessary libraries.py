# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Step 1: Read the Excel file into a DataFrame
file_path = 'C:/Users/etzelcr16/Downloads/baseball.xlsx'
data = pd.read_excel(file_path)

# Step 2: Data Preprocessing

# 2.1: Selecting relevant columns for prediction and target
# Columns D to I contain the features, and column J contains the target variable
features = data.iloc[:, 3:9]  # Columns D to I
target = data['Playoffs']  # Column J

# 2.2: Splitting the data into training and testing sets
# We will use 80% of the data for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Step 3: Model Training

# 3.1: Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3.2: Initializing and training a logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Step 4: Model Evaluation

# 4.1: Predicting on the test set
predictions = model.predict(X_test_scaled)

# 4.2: Calculating accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy of the model:", accuracy)

# Step 5: Making Predictions

# Predicting playoff outcome for each team in the test set
test_predictions = model.predict(X_test_scaled)
print("Predicted playoff outcomes for the test set:")
print(test_predictions)

# Printing actual playoff outcomes for comparison
print("Actual playoff outcomes for the test set:")
print(y_test.values)

# Assuming new data for prediction (You can replace these values with actual data)
new_data = [[750, 650, 90, 0.350, 0.450, 0.280],  # Example team statistics
            [600, 700, 80, 0.320, 0.400, 0.270]]  # Another example team statistics

# Standardizing the new data
new_data_scaled = scaler.transform(new_data)

# Predicting playoff status for new data
predicted_playoffs = model.predict(new_data_scaled)
print("Predicted playoffs status for new data:", predicted_playoffs)

print("Go Brewers")