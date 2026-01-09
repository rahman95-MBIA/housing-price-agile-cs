
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib

df = pd.read_csv("Housing.csv")

X = df[["area", "bedrooms", "bathrooms", "stories", "mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "parking", "prefarea", "furnishingstatus"]]
y = df["price"]

preprocess = ColumnTransformer([
    ("onehot", OneHotEncoder(handle_unknown="ignore"), ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning","prefarea", "furnishingstatus"])
], remainder="passthrough")

model_v2 = Pipeline([
    ("preprocess", preprocess),
    ("regressor", LinearRegression())
])

model_v2.fit(X, y)

joblib.dump(model_v2, "housing_price_model_v2.pkl")
print("Improved model saved.")
