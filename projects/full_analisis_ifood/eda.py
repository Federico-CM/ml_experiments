"""
Script to perform  an exploratory data analysis
Each section highlights features or issues of the data
See the report for a detailed explanation

Make sure that the script is in the same location as ml_project1_data.csv
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind

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

print("\nData range check:")
print(df.select_dtypes(include='number').agg(['min', 'max']).T)

# -------------------
# Data engineering
# -------------------

# Total Spend
sales_cols = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts"
]
df["TotalSpend"] = df[sales_cols].sum(axis=1)

# Total Purchases
purchases_cols = [
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases"
]
df["TotalPurchases"] = df[purchases_cols].sum(axis=1)

# Accepted any previous campaign
campaign_cols = [
    "AcceptedCmp1",
    "AcceptedCmp2",
    "AcceptedCmp3",
    "AcceptedCmp4",
    "AcceptedCmp5"
]
df['AcceptedAnyPrev'] = (df[campaign_cols].sum(axis=1) > 0).astype(int)


# -------------------
# 1- Missing values
# -------------------
print("\nMissing values (%):")
print((df.isna().mean() * 100).sort_values(ascending=False))

test_cols = [
    "Year_Birth",
    "Recency",
    "TotalSpend",
    "TotalPurchases",
]

results = (
    df[test_cols]
    .apply(lambda col: ttest_ind(
        col[df["Income_missing"] == 0],
        col[df["Income_missing"] == 1],
        nan_policy="omit"
    )[1])
)
print("\nt-test association of NA values:")
print(results)


# -------------------
# 2- Constant columns
# -------------------
print("\nColumns with only 1 unique value:")
print(df.nunique()[df.nunique() == 1])


# -------------------
# 3- Outlier detection (IQR method)
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
# 4- Categorical inspection
# -------------------
print("\nEducation categories:")
print(df["Education"].value_counts())

print("\nMarital_Status categories:")
print(df["Marital_Status"].value_counts())


# -------------------
# 5- Campaign acceptance distribution
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
# 6- Sales vs Campaign response
# -------------------

plt.figure()
df.boxplot(column="TotalSpend", by="Response")
plt.title("Total Spend vs Campaign Response")
plt.suptitle("")  # removes automatic pandas title
plt.xlabel("Response")
plt.ylabel("Total Spend")
plt.show()


# -------------------
# 7- Total purchases vs Campaign response
# -------------------

plt.figure()
df.boxplot(column="TotalPurchases", by="Response")
plt.title("Total Purchases vs Campaign Response")
plt.suptitle("")  # removes automatic pandas title
plt.xlabel("Response")
plt.ylabel("Total Purchases")
plt.show()


# -------------------
# 8- Past campaign analysis
# -------------------

print("Previous campaign acceptance vs. response cross-tabulation:")
print(pd.crosstab(df['AcceptedAnyPrev'], df['Response'], normalize='index'))


# -------------------
# 9- Age analysis (by decade cohorts)
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
