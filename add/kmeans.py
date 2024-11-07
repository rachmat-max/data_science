import mysql.connector
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Connect to MySQL Database
db_connection = mysql.connector.connect(
    host='localhost',  # Change to your host
    user='root',       # Your MySQL username
    password='',  # Your MySQL password
    database='your_database'  # Your database name
)

# Step 2: Query the data from the database
query = "SELECT Feature1, Feature2, Feature3 FROM example_data;"  # Select relevant columns
df = pd.read_sql(query, con=db_connection)

# Step 3: Inspect the data
print("Dataset:")
print(df.head())  # Check the first few rows of the data

# Step 4: Preprocess the data (optional but recommended)
# Normalize the data before clustering
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)

# Step 5: Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # Adjust number of clusters
df['Cluster'] = kmeans.fit_predict(data_scaled)

# Step 6: View the results
print("\nClustered Data:")
print(df.head())

# Step 7: Close the database connection
db_connection.close()
