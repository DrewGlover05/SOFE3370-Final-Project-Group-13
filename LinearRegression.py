# SOFE 3370 Final Project
# Battery Pack SOH Prediction using Linear Regression
# October 12th 2025

# Import all of the libraries you need to run the program
import pandas as pd # Used to handle and analyze the data
import numpy as np # Used for mathematical operations
import matplotlib.pyplot as plt # Used for plotting the graph
from sklearn.linear_model import LinearRegression # Linear regression model
from sklearn.model_selection import train_test_split # Splits data into training and testing
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error # Evaluates the performance

# Load dataset into a pandas DataFrame
# The Excel file needs to be in the same folder as this script
df = pd.read_excel("Final Project\\PulseBat Dataset.xlsx")

# Convert DataFrame to a list of lists for Python sorting
data_list = df.values.tolist()

# Sort the data by SOH using merge sort
print("Sorting data by SOH using merge sort...")
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        # Compare by SOH
        if left[i][-1] <= right[j][-1]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

print("Sorting completed.\n")

# Sort the data by SOH
sorted_data = merge_sort(data_list)

# Convert back to a DataFrame
columns = df.columns.tolist()
df_sorted = pd.DataFrame(sorted_data, columns=columns)

# Select which columns will be used as inputs (U1-21)
# The target value we are trying to predict is the battery pack's SOH
X = df_sorted[['U1','U2','U3','U4','U5','U6','U7','U8','U9','U10',
               'U11','U12','U13','U14','U15','U16','U17','U18','U19','U20','U21']]
y = df_sorted['SOH']

# Use train_test_split to split data into 80% training and 20% testing
# random_state=1 uses the same training and testing pattern so the results are consistent
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle=True)

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

# Ask the user for a threshold
threshold = float(input("Enter SOH threshold value: "))
# Use the trained model to predict SOH for the dataset
df_sorted['Predicted_SOH'] = model.predict(X)
# Create a new column to show if the battery passes the threshold
df_sorted['Battery_Status'] = np.where(df_sorted['Predicted_SOH'] < threshold, "The battery has a problem", "The battery is healthy")

# Print the first 10 predictions to see how the model classified them
print("\nPredictions of the first 10 batteries:")
print(df_sorted[['SOH','Predicted_SOH','Battery_Status']].head(10))

# Count how many batteries are healthy vs unhealthy
healthy_count = df_sorted[df_sorted['Battery_Status'] == "The battery is healthy"].shape[0]
unhealthy_count = df_sorted[df_sorted['Battery_Status'] == "The battery has a problem"].shape[0]

# Print the total number of batteries above and below the threshold
print("\nNumber of healthy batteries:", healthy_count)
print("Number of unhealthy batteries:", unhealthy_count)

# Create a scatter plot that compares the predicted SOH to actual SOH
plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual SOH")
plt.ylabel("Predicted SOH")
plt.title("Linear Regression of Battery Pack SOH Prediction")
plt.show()


