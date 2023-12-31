# Sonar Rock vs Mine Prediction

This project involves using a machine learning model to classify sonar signals as either rocks (R) or mines (M).

## Dependencies

Make sure you have the following dependencies installed:

- Python 3.x
- NumPy
- pandas
- scikit-learn

## Data Collection & Processing

- The dataset is loaded from the file `sonar_data.csv`.
- The dataset contains 208 instances, each with 60 features.
- Summary statistics and counts of rock and mine instances are provided.

## Training the Model

- A Logistic Regression model is used for classification.
- The dataset is split into training and test sets (90% training, 10% test).
- The model is trained on the training data and evaluated on both training and test sets.

```python
# Example usage
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ... (data loading and processing)

# Splitting the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

# Training the model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Evaluating the model
train_accuracy = accuracy_score(model.predict(X_train), Y_train)
test_accuracy = accuracy_score(model.predict(X_test), Y_test)

print(f'Accuracy on Training Data: {train_accuracy}')
print(f'Accuracy on Test Data: {test_accuracy}')
