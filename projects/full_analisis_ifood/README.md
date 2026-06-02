# iFood Customer Data Analysis

## What is this?
This project contains a complete analysis of the iFood customer marketing dataset.

The objective is to explore the data, identify quality issues, prepare the dataset for modeling, and build a logistic regression model capable of predicting customer responses to marketing campaigns.

The project is divided into three main stages:

1. **Exploratory Data Analysis (EDA)** (`eda.py`)
2. **Data Processing and Cleaning** (`process_data.py`)
3. **Logistic Regression Analysis** (`logreg.py`)

The findings from each stage are documented in dedicated reports.

## Why is this interesting?
Real-world datasets are rarely clean and ready for machine learning.

Before building predictive models, it is important to understand:

- Missing values
- Outliers
- Inconsistent data
- Feature distributions
- Relationships between variables
- Potential sources of bias

This project demonstrates a realistic machine learning workflow, where data quality assessment and preprocessing are essential steps before modeling.

Similar approaches are used in:

- Marketing campaign optimization
- Customer retention analysis
- Customer behavior prediction
- Business intelligence
- Data-driven decision making

## What does this project show?
This project is an example of:

- How to perform exploratory data analysis on a real-world dataset
- How to identify and document data quality issues
- How to create visualizations that reveal patterns and anomalies
- How to preprocess data for machine learning
- How logistic regression can be used for binary classification
- How model results can be interpreted and evaluated

# Project Structure

```
.
├── eda.py
├── eda_report.md
├── process_data.py
├── logreg.py
├── logreg_report.md
└── data/
    └── ifood.csv
```

## Exploratory Data Analysis

The script `eda.py` performs a series of analyses and validation checks on the dataset.

These include:

- Missing value detection
- Distribution analysis
- Outlier identification
- Correlation analysis
- Feature visualization
- Consistency checks

The generated visualizations help identify potential problems in the data and provide insight into customer behavior.

All findings are summarized in:

`eda_report.md`

## Data Processing

After the exploratory analysis, the dataset can be cleaned and transformed using:

`process_data.py`

This script prepares the data for machine learning by applying the necessary preprocessing steps identified during the EDA phase.

Typical processing tasks may include:

- Handling missing values
- Removing invalid records
- Feature engineering
- Encoding categorical variables
- Scaling or transforming features

The processed dataset is then used as input for the logistic regression analysis.

## Logistic Regression Analysis

The script:

`logreg.py`

trains and evaluates a logistic regression model using the processed dataset.

The objective is to predict whether a customer is likely to respond positively to a marketing campaign.

The model analysis includes:

- Feature importance interpretation
- Model performance evaluation
- Classification metrics
- Discussion of results and limitations

Results and conclusions are documented in:

`logreg_report.md`

# Technical Stuff

## How do I execute the code?

Install the required libraries:

```bash
pip3 install pandas numpy matplotlib seaborn scikit-learn
```

Run the exploratory analysis first:

```bash
python3 eda.py
```

Review the findings in:

```text
eda_report.md
```

Process the data:

```bash
python3 process_data.py
```

Run the logistic regression analysis:

```bash
python3 logreg.py
```

Review the model results in:

```text
logreg_report.md
```

## How do I interpret the results?

The project generates two main reports:

### EDA Report

The EDA report documents:

- Data quality issues
- Missing values
- Outliers
- Distribution characteristics
- Relationships between variables

The goal is to understand the dataset and determine what preprocessing steps are required before modeling.

### Logistic Regression Report

The logistic regression report documents:

- Model performance metrics
- Predictive power of the model
- Important variables influencing predictions
- Strengths and weaknesses of the model

Together, these reports provide a complete view of the dataset, from raw data exploration to predictive modeling.
