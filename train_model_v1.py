import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv("Housing.csv")

X = df[["area"]]
y = df["price"]

model_v1 = LinearRegression()
model_v1.fit(X, y)

joblib.dump(model_v1, "housing_price_model_v1.pkl")
print("Baseline model saved.")
