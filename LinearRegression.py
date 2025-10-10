# SOFE 3370 Final Project
# Battery Pack SOH Prediction using Linear Regression
# October 12 2025

# Import all of the libraries you need to run the program
import pandas as pd # Used to handle and analyze the data
import numpy as np # Used for mathematical operations
import matplotlib.pyplot as plt # Used for plotting the graph
from sklearn.linear_model import LinearRegression # Linear regression model
from sklearn.model_selection import train_test_split # Splits data into training and testing
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error # Evaluates the performance

# Load dataset into a pandas DataFrame
# The Excel file needs to be in the same folder as this script
df = pd.read_excel("PulseBat Dataset.xlsx")

# Select which columns will be used as inputs (U1-21)
# The target value we are trying to predict is the battery pack's SOH
X = df[['U1','U2','U3','U4','U5','U6','U7','U8','U9','U10',
        'U11','U12','U13','U14','U15','U16','U17','U18','U19','U20','U21']]
y = df['SOH']

# Use train_test_split to split data into 80% training and 20% testing
# random_state=1 uses the same training and testing pattern so the results are consistent
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# Create an instance of the Linear Regression Model
model = LinearRegression()
# Fit it using the training data
model.fit(X_train, y_train)

# Use the trained model to predict SOH values for the test data set
y_pred = model.predict(X_test)

# Evaluate how well the model performed
# Use the metrics specified in the guidelines (R^2, MSE, and MAE)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Print out the model performance results
print("Model Evaluation Results:")
print("R2 Score:", round(r2, 4))
print("Mean Squared Error:", round(mse, 6))
print("Mean Absolute Error:", round(mae, 6), "\n")


# Ask the user to input a threshold
threshold = float(input("Enter SOH threshold value: "))
# Use the trained model to predict SOH for the dataset
df['Predicted_SOH'] = model.predict(X)
# Create a new column to show if the battery passes the threshold
df['Battery_Status'] = np.where(df['Predicted_SOH'] < threshold, "The battery has a problem", "The battery is healthy")

# Print the first 10 predictions to see how the model classified them
print("\nPredictions of the first 10 batteries:")
print(df[['SOH', 'Predicted_SOH', 'Battery_Status']].head(10))

# Count how many batteries are healthy vs unhealthy
healthy_count = df[df['Battery_Status'] == "The battery is healthy"].shape[0]
unhealthy_count = df[df['Battery_Status'] == "The battery has a problem"].shape[0]

# Print the total number of batteries above and below the threshold
print("\nNumber of healthy batteries:", healthy_count)
print("Number of unhealthy batteries:", unhealthy_count)