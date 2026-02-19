# Breast Cancer Wisconsin Dataset – Logistic Regression (Python and R)

## What is this?
This project contains two implementations (Python and R) that perform Logistic Regression using the fameous Wisconsin Breast Cancer Dataset (Diagnostic version).

The goal is simple:
Given measurements computed from a breast mass (such as radius, texture, smoothness, etc.), can we correctly predict whether the tumor is malignant (cancerous) or benign (non-cancerous)?

## Why is this interesting?
This project demonstrates how a computer can assist in medical decision-making by learning patterns from real diagnostic data.
This type of analisis is also used for problems that require to identify data in two categories or that have True/False answers.

Examples:
- Customer churn prediction
- Fraud detection
- Spam filtering
- Loan default prediction

## What does this project show?
This project is an example of:
- Binary classification using Logistic Regression
- How to estimate probabilities for medical outcomes
- How to train and test a predictive model
- How to evaluate classification performance
- Implementing the same analysis in both Python and R

# Technical Stuff

## How do I execute the code?
Depending on which script you choose, you just need to execute it in python or R.
If you execute the python script you need the sklearn library before executing the script.
To get that library type: 
pip3 install sklearn

If you execute the R script you need the caret and glmnet libraries before executing the script.
To get those libraries, type:
install.packages(c("caret","glmnet"))

## How do I interpret the results?
The models present a lot of valuable diagnostics.
Among them, two pieces of information are easily interpretable:

1) The confusion matrix
It should look something like this:

                 Predicted
               Malignant   Benign
Actual
Malignant           53         1
Benign               0        89

In medical contexts, minimizing false negatives (predicting benign when it is malignant) is especially important.


