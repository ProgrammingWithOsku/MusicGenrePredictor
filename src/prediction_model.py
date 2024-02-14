import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Load the dataset
music_data = pd.read_csv('../data/music.csv')
X = music_data.drop(columns=['genre'])
y = music_data['genre']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize and train the Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Calculate and print the accuracy score
score = accuracy_score(y_test, predictions)
print(f"Accuracy score: {score}")

# Plot the decision tree
plt.figure(figsize=(20,10))
plot_tree(model, feature_names=X.columns.tolist(), class_names=y.unique().tolist(), filled=True, rounded=True)
plt.title("Decision Tree for Music Genre Prediction")
plt.show()
