import mysql.connector
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

# Step 1: Connect to MySQL database and fetch data
def fetch_data_from_mysql():
    # Replace with your actual MySQL database credentials
    conn = mysql.connector.connect(
        host="localhost",      # MySQL server hostname
        user="root",           # Your MySQL username
        password="",   # Your MySQL password
        database="your_database_name"  # Your MySQL database
    )
    
    query = "SELECT date, sales FROM sales_data ORDER BY date ASC"
    df = pd.read_sql(query, conn)
    conn.close()
    
    # Ensure the 'date' column is in datetime format
    df['date'] = pd.to_datetime(df['date'])
    
    return df

# Step 2: Apply Exponential Smoothing
def apply_exponential_smoothing(df):
    # Set 'date' as the index
    df.set_index('date', inplace=True)
    
    # Apply Exponential Smoothing model
    model = ExponentialSmoothing(df['sales'], trend='add', seasonal=None, damped_trend=False, seasonal_periods=12)
    fit = model.fit(smoothing_level=0.8, optimized=False)
    
    # Forecast the next 5 periods (optional)
    forecast = fit.forecast(steps=5)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['sales'], label='Observed Sales')
    plt.plot(df.index, fit.fittedvalues, label='Fitted Values', linestyle='--')
    plt.plot(forecast.index, forecast, label='Forecast', linestyle=':', color='red')
    plt.title("Exponential Smoothing")
    plt.legend()
    plt.show()
    
    return fit, forecast

# Main execution flow
if __name__ == "__main__":
    # Step 1: Fetch the data
    df = fetch_data_from_mysql()
    
    # Step 2: Apply Exponential Smoothing
    fit, forecast = apply_exponential_smoothing(df)
    
    # Output forecast
    print("Forecasted values for the next 5 days:")
    print(forecast)

# ------------------------------------------For Data Example-----------------------------------
# CREATE DATABASE your_database_name;
# USE your_database_name;

# CREATE TABLE sales_data (
#     id INT AUTO_INCREMENT PRIMARY KEY,
#     date DATE,
#     sales DECIMAL(10, 2)
# );

# INSERT INTO sales_data (date, sales) VALUES
# ('2023-01-01', 100.00),
# ('2023-01-02', 110.00),
# ('2023-01-03', 120.00),
# ('2023-01-04', 130.00),
# ('2023-01-05', 140.00);
