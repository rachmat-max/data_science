import mysql.connector
import numpy as np
import pandas as pd

# Step 1: Connect to MySQL database and fetch data
def get_data_from_mysql():
    # MySQL database connection details
    connection = mysql.connector.connect(
        host='localhost',  # Your MySQL host
        user='root',       # Your MySQL username
        password='',       # Your MySQL password
        database='saw'     # Your database name
    )
    
    # SQL query to retrieve the data
    query = "SELECT * FROM alternatives;"
    
    # Read data into a Pandas DataFrame
    df = pd.read_sql(query, connection)
    
    # Close the connection
    connection.close()
    
    return df

# Step 2: Normalize the data (min-max normalization)
def normalize_data(df):
    # Normalize each criterion using min-max normalization
    criteria_columns = df.columns[1:]  # Assuming the first column is 'alternative_name'
    
    for column in criteria_columns:  # Skip the 'alternative_name' column
        min_val = df[column].min()
        max_val = df[column].max()
        df[column] = (df[column] - min_val) / (max_val - min_val)
    
    return df

# Step 3: Assign weights to criteria
def assign_weights(df):
    # Dynamically create the weights based on the column names
    criteria_columns = df.columns[1:]  # Assuming the first column is 'alternative_name'
    
    # Example weights for the criteria (You may change these based on your use case)
    weights = {col: 1/len(criteria_columns) for col in criteria_columns}
    
    return weights

# Step 4: Compute SAW scores
def compute_saw_scores(df, weights):
    # Calculate the weighted sum for each alternative
    scores = []
    criteria_columns = df.columns[1:]  # Skip the 'alternative_name' column
    
    for index, row in df.iterrows():
        score = 0
        for criterion, weight in weights.items():
            score += row[criterion] * weight
        scores.append(score)
    
    df['score'] = scores
    return df

# Step 5: Rank alternatives based on their scores
def rank_alternatives(df):
    df['rank'] = df['score'].rank(ascending=False, method='min')
    return df.sort_values(by='rank')

# Main function to run the SAW method
def main():
    # Step 1: Get data from MySQL
    df = get_data_from_mysql()
    
    # Step 2: Normalize the data
    df = normalize_data(df)
    
    # Step 3: Assign weights to criteria
    weights = assign_weights(df)
    
    # Step 4: Compute SAW scores
    df = compute_saw_scores(df, weights)
    
    # Step 5: Rank the alternatives
    ranked_df = rank_alternatives(df)
    
    # Display the results
    print(ranked_df[['alternative_name', 'score', 'rank']])

if __name__ == '__main__':
    main()
