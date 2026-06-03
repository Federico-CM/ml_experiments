# iFood Customer Data Analysis

## What is this?
This project contains a complete analysis of the iFood customer marketing dataset.

The objective is to explore the data, identify quality issues, prepare the dataset for modeling, and build a logistic regression model capable of predicting customer responses to marketing campaigns.

The project is divided into three main stages:

1. **Exploratory Data Analysis (EDA)** (`eda.py`)
2. **Data Processing and Cleaning** (`process_data.py`)
3. **Modelling** (`logreg.py` & `r_forest.py`)

The findings from each stage are documented in dedicated reports ([1](https://github.com/Federico-CM/ml_experiments/blob/main/projects/full_analysis_ifood/reports/eda_report.pdf),[2](https://github.com/Federico-CM/ml_experiments/blob/main/projects/full_analysis_ifood/reports/data_processing_report.pdf),[3](https://github.com/Federico-CM/ml_experiments/blob/main/projects/full_analysis_ifood/reports/modelling_report.pdf))

## Why is this interesting?
This project uses the iFood Customer dataset, originally designed as a hiring case study for the iFood Brain Data Analytics team.

The dataset contains realistic data quality issues and imperfections, making it a good example of a real-world analytics workflow: exploratory analysis, data cleaning, feature preparation, and predictive modeling.

## What does this project show?
This project is an example of:

- How to perform exploratory data analysis on a real-world dataset
- How to identify and document data quality issues
- How to create visualizations that reveal patterns and anomalies
- How to preprocess data for machine learning
- How logistic regression can be used for binary classification
- How model results can be interpreted and evaluated

## Exploratory Data Analysis

The script `eda.py` performs a series of analyses and validation checks on the dataset.

These include:

- Missing value detection
- Distribution analysis
- Outlier identification
- Feature visualization
- Consistency checks

The generated visualizations help identify potential problems in the data and provide insight into customer behavior.

All findings are summarized in:

[eda_report.pdf](https://github.com/Federico-CM/ml_experiments/blob/main/projects/full_analysis_ifood/reports/eda_report.pdf)

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

All changes are summarized in:

[data_processing_report.pdf](https://github.com/Federico-CM/ml_experiments/blob/main/projects/full_analysis_ifood/reports/data_processing_report.pdf)

## Logistic Regression Analysis

The scripts:

`logreg.py` and `r_forest.py`

trains and evaluates a logistic regression model and a random forest model using the processed dataset.

The objective is to predict whether a customer is likely to respond positively to a marketing campaign.

The model analysis includes:

- Feature importance interpretation
- Model performance evaluation
- Classification metrics
- Discussion of results and limitations

Results and conclusions are documented in:

[modelling_report.pdf](https://github.com/Federico-CM/ml_experiments/blob/main/projects/full_analysis_ifood/reports/modelling_report.pdf)

# Technical Stuff

## How do I execute the code?

Install the required libraries:

```bash
pip3 install pandas numpy matplotlib seaborn scikit-learn
```


Run the logistic regression model:

```bash
python3 logreg.py
```

Run the random forest model:

```bash
python3 r_forest.py
```

## How do I interpret the results?

The models generate several sumary statistics.
An interpretation is offered in the [modelling_report.pdf](https://github.com/Federico-CM/ml_experiments/blob/main/projects/full_analysis_ifood/reports/modelling_report.pdf).
Accuracy measures overall correctness, ROC AUC evaluates the model’s ability to distinguish classes, and PR AUC is especially useful for imbalanced datasets. Precision indicates how reliable positive predictions are, while recall measures how many actual positives are detected. The confusion matrix shows the number of correct and incorrect predictions, and cross-validation results assess the model’s stability and generalization performance.
