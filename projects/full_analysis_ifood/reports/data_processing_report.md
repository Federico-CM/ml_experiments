# Data Processing Report

## Summary
The dataset undergoes a series of preprocessing steps designed to address known data quality issues identified during exploratory analysis.

**The processing pipeline removes uninformative variables, handles missing and invalid values, standardizes date fields, and transforms income data for improved analytical suitability.**

The resulting dataset is cleaner, contains fewer data quality issues, and is better suited for downstream statistical modeling and machine learning tasks.

All preprocessing is performed programmatically through a dedicated data processing script. The raw dataset is retained as the single source of truth, and no intermediate processed files are generated. The preprocessing pipeline is executed whenever data are prepared for modeling or analysis, ensuring consistency and reproducibility across all modeling activities.

## Processing Steps

### 1. Removal of Uninformative Columns
**Three columns are removed from the dataset:**
- `Z_CostContact`
- `Z_Revenue`
- `ID`

The variables `Z_CostContact` and `Z_Revenue` were previously identified as containing constant values and therefore provide no predictive information.

The `ID` column serves only as a unique identifier and does not contain meaningful information for analysis or modeling.

---

### 2. Missing Value Imputation for Income
**Missing values in `Income` are replaced using median imputation.**

The median is a robust measure of central tendency and is less sensitive to extreme values than the mean. This approach preserves all observations while minimizing the influence of skewed income distributions.

After this step, no missing values remain in the `Income` variable.

---

### 3. Treatment of Implausible Birth Years
**Implausible birth years between 1893 and 1900 are treated as invalid values.**

Records with birth years within this range are first recoded as missing values and subsequently removed from the dataset.

This process eliminates observations that likely originate from data entry errors while retaining records with plausible demographic information.

---

### 4. Cleaning of Marital Status Categories
**Invalid categories in `Marital_Status` are removed.**

The following values are considered invalid:
- `Absurd`
- `YOLO`

Observations containing these values are recoded as missing and subsequently removed from the dataset.

This ensures that the marital status variable contains only meaningful and interpretable categories.

---

### 5. Date Parsing and Standardization
**The `Dt_Customer` column is converted to a datetime format.**

Values that cannot be successfully parsed are converted to missing values.

Standardizing dates facilitates subsequent temporal analyses and feature engineering activities based on customer enrollment dates.

---

### 6. Income Transformation
**A logarithmic transformation is applied to the income variable.**

A new variable, `Income_log`, is created using:

\[
Income\_log = \log(1 + Income)
\]

The transformation reduces the influence of extreme income values, decreases skewness, and often improves the performance of statistical and machine learning models.

After the transformation is created, the original `Income` column is removed from the dataset.

---

## Resulting Dataset
After processing:

- Uninformative columns have been removed.
- Missing income values have been imputed using the median.
- Records containing implausible birth years have been excluded.
- Records containing invalid marital status values have been excluded.
- Customer dates have been standardized to datetime format.
- Income has been transformed to a log scale and replaced by `Income_log`.

The resulting dataset contains only cleaned and transformed variables intended for subsequent analysis and modeling.

## Conclusion
The preprocessing pipeline addresses the primary data quality issues identified during exploratory analysis while preserving as much information as possible.

The combination of median imputation, removal of invalid records, date standardization, and logarithmic income transformation produces a dataset that is more robust for statistical analysis and predictive modeling.

Particular attention should be given to interpreting the transformed income variable, as all future analyses involving income should use `Income_log` rather than the original income scale.

Because the preprocessing is implemented as a reusable script and applied directly to the raw data when required, the workflow remains reproducible and maintainable while preserving a single authoritative source of data. 
