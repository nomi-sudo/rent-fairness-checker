import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

df_raw = pd.read_csv("House_Rent_Dataset.csv")
df = df_raw.rename(columns={
    "City": "city", "BHK": "bhk", "Size": "size_sqft",
    "Furnishing Status": "furnishing", "Rent": "rent"
})[["city", "bhk", "size_sqft", "furnishing", "rent"]].copy()

print("Dataset shape:", df.shape)
print(df.head())
print("\nAverage rent by city:")
print(df.groupby("city")["rent"].mean().round(0).sort_values(ascending=False))

df_model = pd.get_dummies(df, columns=["city", "furnishing"], drop_first=True)
X = df_model.drop(columns="rent")
y = df_model["rent"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7)

model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)

print(f"\nModel R^2: {r2_score(y_test, preds):.3f}")
print(f"Model MAE: {mean_absolute_error(y_test, preds):.0f}")

df["predicted_fair_rent"] = model.predict(df_model.drop(columns="rent"))
df["price_gap_pct"] = ((df["rent"] - df["predicted_fair_rent"])
                       / df["predicted_fair_rent"] * 100).round(1)

def flag(gap):
    if gap > 15:
        return "OVERPRICED"
    elif gap < -15:
        return "UNDERPRICED (good deal)"
    return "fair"

df["verdict"] = df["price_gap_pct"].apply(flag)

print("\nSample listings with fairness verdict:")
print(df[["city", "bhk", "size_sqft", "rent", "predicted_fair_rent",
          "price_gap_pct", "verdict"]].head(10).to_string(index=False))

over = (df["verdict"] == "OVERPRICED").sum()
under = (df["verdict"] == "UNDERPRICED (good deal)").sum()
fair = (df["verdict"] == "fair").sum()
print(f"\nOverpriced listings: {over} ({over / len(df) * 100:.1f}%)")
print(f"Good deals found:    {under} ({under / len(df) * 100:.1f}%)")
print(f"Fair listings:       {fair} ({fair / len(df) * 100:.1f}%)")

city_avg = df.groupby("city")["rent"].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
bars = plt.bar(city_avg.index, city_avg.values, color="#2b6cb0")
for bar, val in zip(bars, city_avg.values):
    plt.text(bar.get_x() + bar.get_width() / 2, val + 1800, f"Rs {val:,.0f}",
             ha="center", fontsize=12, fontweight="bold")
plt.ylabel("Average rent (Rs / month)", fontsize=12)
plt.title("Average Rent by City", fontsize=15, fontweight="bold")
plt.ylim(0, city_avg.max() * 1.15)
plt.tight_layout()
plt.savefig("avg_rent_by_city.png", dpi=200)
plt.close()

df.to_csv("projectC_output_data.csv", index=False)
print("\nSaved: projectC_output_data.csv")
print("Graph saved: avg_rent_by_city.png")
