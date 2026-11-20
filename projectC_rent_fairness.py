# Project C: Rent Price Fairness Checker
# Dataset: House Rent Prediction Dataset - kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_csv("House_Rent_Dataset.csv")
df = df.rename(columns={
    "Size": "size_sqft",
    "City": "city",
    "Furnishing Status": "furnishing",
    "Rent": "rent",
    "BHK": "bhk",
})

print("Dataset shape:", df.shape)
print(df[["city", "bhk", "size_sqft", "furnishing", "rent"]].head())

print("\nAverage rent by city:")
print(df.groupby("city")["rent"].mean().round(0).sort_values(ascending=False))

model_cols = ["bhk", "size_sqft", "city", "furnishing", "rent"]
df_model = pd.get_dummies(df[model_cols], columns=["city", "furnishing"], drop_first=True)

X = df_model.drop(columns="rent")
y = df_model["rent"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)

print(f"\nModel R^2: {r2_score(y_test, preds):.3f}")
print(f"Model MAE: {mean_absolute_error(y_test, preds):.0f}")

df["predicted_fair_rent"] = model.predict(X)
df["price_gap_pct"] = ((df["rent"] - df["predicted_fair_rent"]) / df["predicted_fair_rent"] * 100).round(1)

def flag(gap):
    if gap > 15:
        return "OVERPRICED"
    elif gap < -15:
        return "UNDERPRICED (good deal)"
    return "fair"

df["verdict"] = df["price_gap_pct"].apply(flag)

print("\nSample listings with fairness verdict:")
print(df[["city", "bhk", "size_sqft", "rent", "predicted_fair_rent", "price_gap_pct", "verdict"]]
      .head(10).to_string(index=False))

print(f"\nOverpriced listings: {(df['verdict']=='OVERPRICED').sum()} "
      f"({(df['verdict']=='OVERPRICED').mean()*100:.1f}%)")
print(f"Good deals found:    {(df['verdict']=='UNDERPRICED (good deal)').sum()} "
      f"({(df['verdict']=='UNDERPRICED (good deal)').mean()*100:.1f}%)")
print(f"Fair listings:       {(df['verdict']=='fair').sum()} "
      f"({(df['verdict']=='fair').mean()*100:.1f}%)")

df.to_csv("projectC_output_data.csv", index=False)
print("\nSaved: projectC_output_data.csv")

# Scatter: actual vs. predicted rent, color-coded by verdict
colors = {"OVERPRICED": "#e74c3c", "fair": "#95a5a6", "UNDERPRICED (good deal)": "#2ecc71"}
view_limit = df["rent"].quantile(0.99)  # clip so outlier listings don't flatten the chart

plt.figure(figsize=(8, 7))
for verdict, color in colors.items():
    subset = df[df["verdict"] == verdict]
    plt.scatter(subset["predicted_fair_rent"], subset["rent"],
                s=12, alpha=0.5, label=verdict, color=color)

plt.plot([0, view_limit], [0, view_limit], "k--", linewidth=1, label="Perfectly fair (y = x)")
plt.xlim(0, view_limit)
plt.ylim(0, view_limit)

plt.xlabel("Model-predicted fair rent (₹)")
plt.ylabel("Actual listed rent (₹)")
plt.title("Rent Fairness Checker: Actual vs. Predicted Rent")
plt.legend()
plt.tight_layout()
plt.savefig("rent_fairness_scatter.png", dpi=150)
plt.close()
print("\nSaved chart to rent_fairness_scatter.png")
