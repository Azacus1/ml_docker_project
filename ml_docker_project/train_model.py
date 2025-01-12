from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset and train model
path = "./model.pkl"
iris = load_iris()
clf = RandomForestClassifier()
clf.fit(iris.data, iris.target)

# Save the trained model
joblib.dump(clf, path)
print("Model trained and saved as model.pkl")