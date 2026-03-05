"""
Script to perform  an exploratory data analysis
Each section highlights features or issues of the data
See the report for a detailed explanation

Make sure that the script is in the same location as ml_project1_data.csv
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# -------------------
# Load  the data
# -------------------
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "ml_project1_data.csv"
df = pd.read_csv(CSV_PATH)
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], errors="coerce")

# -------------------
# Basic sanity check
# -------------------
print("Shape:", df.shape)
print("\nData types:\n", df.dtypes)

# -------------------
# Missing values
# -------------------
print("\nMissing values (%):")
print((df.isna().mean() * 100).sort_values(ascending=False))

# -------------------
# Constant columns
# -------------------
print("\nColumns with only 1 unique value:")
print(df.nunique()[df.nunique() == 1])

# -------------------
# Data range check
# -------------------
print("\nData range check:")
print(df.select_dtypes(include='number').agg(['min', 'max']).T)

# -------------------
# Outlier detection (IQR method)
# -------------------
def detect_outliers(series):
    """A function to get outliers in a column"""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return series[(series < lower) | (series > upper)]

print("\nYear_Birth outliers:")
print(detect_outliers(df["Year_Birth"]))

print("\nIncome outliers:")
print(detect_outliers(df["Income"].dropna()).sort_values(ascending=False).head())

# -------------------
# Categorical inspection
# -------------------
print("\nEducation categories:")
print(df["Education"].value_counts())

print("\nMarital_Status categories:")
print(df["Marital_Status"].value_counts())

# -------------------
# Campaign acceptance distribution
# -------------------
print("\nResponse distribution:")
print(df["Response"].value_counts().sort_index())

plt.figure()
df["Response"].value_counts().sort_index().plot(kind="bar")
plt.title("Distribution of Response")
plt.xlabel("Campaign response")
plt.ylabel("Count")
plt.show()

# -------------------
# Sales vs Campaign response
# -------------------
sales_cols = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts"
]

df["TotalSpend"] = df[sales_cols].sum(axis=1)

plt.figure()
df.boxplot(column="TotalSpend", by="Response")
plt.title("Total Spend vs Campaign Response")
plt.suptitle("")  # removes automatic pandas title
plt.xlabel("Response")
plt.ylabel("Total Spend")
plt.show()

# -------------------
# Total purchases vs Campaign response
# -------------------
purchases_cols = [
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases"
]

df["TotalPurchases"] = df[purchases_cols].sum(axis=1)

plt.figure()
df.boxplot(column="TotalPurchases", by="Response")
plt.title("Total Purchases vs Campaign Response")
plt.suptitle("")  # removes automatic pandas title
plt.xlabel("Response")
plt.ylabel("Total Purchases")
plt.show()

# -------------------
# Past campaign analysis
# -------------------
campaign_cols = [
    "AcceptedCmp1",
    "AcceptedCmp2",
    "AcceptedCmp3",
    "AcceptedCmp4",
    "AcceptedCmp5"
]

df['AcceptedAnyPrev'] = (df[campaign_cols].sum(axis=1) > 0).astype(int)

print("Previous campaign acceptance vs. response cross-tabulation:")
print(pd.crosstab(df['AcceptedAnyPrev'], df['Response'], normalize='index'))

# -------------------
# Age analysis (by decade cohorts)
# -------------------
df["BirthDecade"] = (df["Year_Birth"] // 10) * 10
df[df["Response"] == 1].groupby("BirthDecade").size()

result = (
    df.groupby("BirthDecade")
      .agg(
          clients=("Response", "size"),
          response1=("Response", "sum")
      )
      .reset_index()
)
result["response1_rate"] = result["response1"] / result["clients"]

print("Response rate per decade cohorts:")
print(result)
