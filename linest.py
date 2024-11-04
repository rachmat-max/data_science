import sqlite3

def load_data(db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute("SELECT x, y FROM data_table")
    data = cursor.fetchall()

    conn.close()
    return data

class SimpleLinearRegression:
    def __init__(self):
        self.m = 0  # slope
        self.b = 0  # intercept

    def fit(self, x, y):
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        self.m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        self.b = (sum_y - self.m * sum_x) / n

    def predict(self, x):
        return [self.m * xi + self.b for xi in x]

# Example Usage
if __name__ == "__main__":
    db_name = 'data.db'  # Change this to your database name
    data = load_data(db_name)

    # Separate the data into x and y
    x = [row[0] for row in data]
    y = [row[1] for row in data]

    # Create and fit the model
    model = SimpleLinearRegression()
    model.fit(x, y)

    # Make predictions
    predictions = model.predict(x)

    print("Slope:", model.m)
    print("Intercept:", model.b)
    print("Predictions:", predictions)
