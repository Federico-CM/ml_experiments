# EDA Report

## Summary
The explored dataset (ifood dataset) has data regarding the response to a marketing campaign by different customers.

**Minor issues were identified in the dataset; however, they do not impede further analysis.**

These issues can be addressed through standard preprocessing methods such as data imputation or removal of rows containing data entry errors.

## Important Findings

### 1. Missing Values in Income
**There are 24 missing values in `Income`** out of 2240 observations (~1.07%).

An MCAR test was performed. No statistical evidence that the missingness mechanism departs from Missing Completely At Random (MCAR) was found. *(EDA script, section 1)*.

Given the small proportion of missing values and the typically skewed distribution of income, median imputation is a reasonable and robust approach.

---

### 2. Uninformative Columns
**Two columns, `Z_CostContact` and `Z_Revenue`, are uninformative** *(EDA script, section 2)*.

These columns contain constant values across all observations (zero variance), meaning they provide no information for modeling and should be removed.

---

### 3. Implausible Birth Records
In the date of birth, there are **three entries with implausible birth years between 1893 and 1900** *(EDA script, section 3)*.

These entries likely represent data entry errors and should either be:
- Removed, or
- Recoded as missing values.

Additionally, **one observation shows an extremely high income** relative to the rest of the dataset *(EDA script, section 3)*.  
Since this could be a legitimate but extreme value, removing it is not advisable. Instead, log-transforming the income variable should be considered.

---

### 4. Issues in Categorical Data
There are unusual values and mislabeled entries in the categorical data *(EDA script, section 4)*.

In the column `Marital_Status`, there are:
- **Two entries labeled "Absurd"**
- **Two entries labeled "YOLO"**

These entries likely represent invalid categories and should either be:
- Removed, or
- Recoded as missing values.

---

### 5. Class Imbalance in Response
**The `Response` variable is imbalanced**, with 334 positives out of 2240 records *(EDA script, section 5)*.

When splitting the dataset into training and test sets, stratified sampling should be used to preserve class proportions.

---

### 6. Associations with Response Variable
The exploratory analysis suggests that **several variables exhibit associations with the `Response` variable** *(EDA script sections 6–9)*. This also includes engineered variables.


## Conclusion
The data is suitable for further analysis, provided care is taken to clean and process the data. 

Visualizations suggest potential associations with the response variable that warrant further investigation through statistical modeling.

Logistic regression is recommended as a baseline model as the response variable is binary. More flexible methods (e.g. tree-based models) may also be explored.