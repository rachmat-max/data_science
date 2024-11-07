import mysql.connector
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Establish MySQL connection
db_connection = mysql.connector.connect(
    host="localhost",       # Your host (e.g., localhost)
    user="root",            # Your MySQL username
    password="",  # Your MySQL password
    database="yourdatabase"   # Name of your database
)

# Query to fetch data from MySQL
query = "SELECT sepal_length, sepal_width, petal_length, petal_width, class FROM iris_data"

# Load data into a Pandas DataFrame
data = pd.read_sql(query, db_connection)

# Close the database connection
db_connection.close()

# Features (X) and Target (y)
X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = data['class']

# Convert categorical labels to numeric using LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Naive Bayes classifier
nb_classifier = GaussianNB()

# Train the classifier
nb_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = nb_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Output results
print(f"Result Prediction: {y_pred}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)


#--------------------------------------------------For Data Example-----------------------------------------
# CREATE DATABASE yourdatabase;
# USE yourdatabase;

# CREATE TABLE iris_data (
#     sepal_length FLOAT,
#     sepal_width FLOAT,
#     petal_length FLOAT,
#     petal_width FLOAT,
#     class VARCHAR(50)
# );

# INSERT INTO iris_data (sepal_length, sepal_width, petal_length, petal_width, class)
# VALUES
# (5.1, 3.5, 1.4, 0.2, 'setosa'),
# (4.9, 3.0, 1.4, 0.2, 'setosa'),
# (4.7, 3.2, 1.3, 0.2, 'setosa'),
# (4.6, 3.1, 1.5, 0.2, 'setosa'),
# (5.0, 3.6, 1.4, 0.2, 'setosa'),
# -- More rows follow...
# (7.7, 3.8, 6.7, 2.2, 'virginica'),
# (7.7, 2.6, 6.9, 2.3, 'virginica'),
# (6.3, 3.4, 5.6, 2.4, 'virginica'),
# (6.4, 3.1, 5.5, 1.8, 'virginica'),
# (6.0, 3.0, 5.2, 1.9, 'virginica');