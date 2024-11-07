import mysql.connector
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Connect to MySQL Database
def fetch_data_from_mysql():
    # Establish MySQL connection
    connection = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='mydatabase'
    )

    # Query to fetch the data
    query = "SELECT * FROM example_data"
    
    # Fetch the data into a Pandas DataFrame
    data = pd.read_sql(query, connection)
    
    # Close the connection
    connection.close()
    
    return data

# Step 2: Preprocess the data
def preprocess_data(data):
    # Assume 'target' is the column to predict, and all others are features
    X = data.drop(columns=['target'])
    y = data['target']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Standardize the features (important for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# Step 3: Train the SVM model
def train_svm(X_train, y_train):
    # Initialize the Support Vector Classifier (SVC)
    svm = SVC(kernel='linear')  # You can choose other kernels like 'rbf', 'poly', etc.
    
    # Train the SVM model
    svm.fit(X_train, y_train)
    
    return svm

# Step 4: Evaluate the model
def evaluate_model(svm, X_test, y_test):
    # Make predictions
    y_pred = svm.predict(X_test)

    print(f"Prediction: {y_pred}")
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    
    # Print the classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# Main function to bring everything together
def main():
    # Fetch data from MySQL
    data = fetch_data_from_mysql()
    
    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    # Train the SVM model
    svm_model = train_svm(X_train, y_train)
    
    # Evaluate the model
    evaluate_model(svm_model, X_test, y_test)

if __name__ == "__main__":
    main()
