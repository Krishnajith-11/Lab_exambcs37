import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import json

# Load dataset (use any CSV dataset)
df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv")

# Assign column names
df.columns = [
    "CRIM","ZN","INDUS","CHAS","NOX","RM","AGE",
    "DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"
]

# Features & target
X = df.drop("MEDV", axis=1)
y = df["MEDV"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)

print("MSE:", mse)

# Save model
joblib.dump(model, "model.pkl")

# Save metrics
with open("metrics.json", "w") as f:
    json.dump({"mse": mse}, f)
print("Name: Krishnajith | Roll No: 2022BCS0037")