import sqlite3
import numpy as np
from collections import defaultdict

# Connect to SQLite database
def load_data(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT features, label FROM data")
    rows = cursor.fetchall()
    
    features, labels = zip(*rows)
    
    conn.close()
    return features, labels

# Naive Bayes Classifier
class NaiveBayes:
    def __init__(self):
        self.class_priors = {}
        self.feature_likelihoods = defaultdict(lambda: defaultdict(int))
        self.classes = set()
    
    def fit(self, features, labels):
        total_count = len(labels)
        
        # Calculate class priors and feature likelihoods
        for feature, label in zip(features, labels):
            self.classes.add(label)
            self.class_priors[label] = self.class_priors.get(label, 0) + 1
            
            for word in feature.split():
                self.feature_likelihoods[label][word] += 1
        
        # Convert counts to probabilities
        for label in self.classes:
            self.class_priors[label] /= total_count
            total_word_count = sum(self.feature_likelihoods[label].values())
            for word in self.feature_likelihoods[label]:
                self.feature_likelihoods[label][word] /= total_word_count
    
    def predict(self, feature):
        max_prob = float('-inf')
        best_class = None
        
        for label in self.classes:
            log_prob = np.log(self.class_priors[label])
            for word in feature.split():
                likelihood = self.feature_likelihoods[label].get(word, 0)
                log_prob += np.log(likelihood) if likelihood > 0 else 0
            
            if log_prob > max_prob:
                max_prob = log_prob
                best_class = label
        
        return best_class

# Example usage
if __name__ == "__main__":
    db_path = 'data.db'  # Replace with your database path
    features, labels = load_data(db_path)
    
    model = NaiveBayes()
    model.fit(features, labels)

    # Test with a new feature
    test_feature = "example text to classify"
    prediction = model.predict(test_feature)
    print(f'The predicted class for "{test_feature}" is: {prediction}')
