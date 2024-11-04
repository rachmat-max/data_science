import sqlite3

def load_data(db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute('SELECT name, weight, value FROM criteria')
    data = cursor.fetchall()

    conn.close()

    return data

def saw_algorithm(data):
    total_weight = sum(weight for _, weight, _ in data)

    scores = [(name, (weight / total_weight) * value) for name, weight, value in data]

    final_score = sum(score for _, score in scores)

    return final_score

if __name__ == "__main__":
    db_name = 'data.db'
    data = load_data(db_name)

    final_score = saw_algorithm(data)

    print(f'Final Score: {final_score:.2f}')