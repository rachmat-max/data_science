import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Connect to SQLite database
conn = sqlite3.connect('data.db')  # Replace with your database file
query = "SELECT date, sales FROM data_table ORDER BY date;"  # Adjust your SQL query
df = pd.read_sql_query(query, conn)

# Close the database connection
conn.close()

# Ensure 'date' is a datetime object and set it as the index
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Split data into train and test sets
train_size = int(len(df) * 0.8)
train, test = df.value[:train_size], df.value[train_size:]

# Simple Autoregressive Model Implementation
def autoregression(train, lag=1):
    n = len(train)
    X = np.zeros((n - lag, lag))
    y = np.zeros(n - lag)
    
    for i in range(lag, n):
        X[i - lag] = train[i - lag:i]
        y[i - lag] = train[i]
    
    # Calculate coefficients using the Normal Equation
    X = np.hstack((np.ones((X.shape[0], 1)), X))  # Add intercept
    coeffs = np.linalg.inv(X.T @ X) @ X.T @ y
    
    return coeffs

def predict(coeffs, train, lag, steps):
    predictions = []
    last_values = train[-lag:]

    for _ in range(steps):
        pred = coeffs[0] + np.sum(coeffs[1:] * last_values)
        predictions.append(pred)
        last_values = np.roll(last_values, -1)  # Shift values
        last_values[-1] = pred  # Update last value with prediction

    return predictions

# Fit the autoregressive model
coeffs = autoregression(train.values, lag=1)

# Make predictions
predictions = predict(coeffs, train.values, lag=1, steps=len(test))

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(df.index, df.value, label='Original Data', color='blue')
plt.plot(test.index, predictions, label='Predictions', color='red')
plt.title('Autoregression Predictions')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()
