from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

# Demo model used so the repository works immediately after cloning.
# Replace this file/model.pkl with your own trained model when submitting
# a project based on a different ML task.
data = load_iris()

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)
model.fit(data.data, data.target)

joblib.dump(model, "model.pkl")
print("Saved trained model to model.pkl")
