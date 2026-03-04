 """
 EDA of the data, each section highlights features or issues of the data
 See the EDA report for a detailed explanation

 """

 import pandas as pd
 import numpy as np
 import matplotlib.pyplot as plt

 # -------------------
 # Load
 # -------------------
 df = pd.read_csv("/home/proud/Desktop/work/AI/i_food/ml_project1_data.csv")
 df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], errors="coerce")

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
 df = df.loc[:,df.nunique() != 1]
