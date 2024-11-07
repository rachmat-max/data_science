import mysql.connector
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Connect to MySQL database
db_connection = mysql.connector.connect(
    host="localhost",       # Replace with your MySQL host
    user="root",   # Replace with your MySQL username
    password="", # Replace with your MySQL password
    database="your_database"  # Replace with your database name
)

# Query to fetch data from the MySQL table
query = "SELECT X, Y FROM your_table_name"  # Replace with your table name and column names

# Load data into a pandas DataFrame
df = pd.read_sql(query, db_connection)

# Close the database connection
db_connection.close()

# Show the first few rows of the data
print(df.head())

# Prepare the data (X and Y)
X = df['X'].values.reshape(-1, 1)  # Independent variable
Y = df['Y'].values  # Dependent variable

# Create a linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, Y)

# Print the slope (coefficient) and the intercept
print(f"Slope: {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")

# Make predictions using the model
Y_pred = model.predict(X)

# Visualize the data and the regression line
plt.scatter(X, Y, color='blue', label='Data Points')
plt.plot(X, Y_pred, color='red', label='Regression Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression')
plt.legend()
plt.show()

# Example of making a prediction for a new value of X
new_X = np.array([[10]])  # Replace with your desired X value
predicted_Y = model.predict(new_X)
print(f"Prediction for X = {new_X[0][0]}: Y = {predicted_Y[0]}")


#-------------------------------------------For Data Example---------------------------------
# CREATE DATABASE regression_db;

# USE regression_db;

# CREATE TABLE sales_data (
#     X FLOAT,
#     y FLOAT
# );

# -- Insert some example data
# INSERT INTO sales_data (X, y) VALUES
# (1, 1.2),
# (2, 2.3),
# (3, 3.5),
# (4, 4.7),
# (5, 5.8),
# (6, 7.0);

