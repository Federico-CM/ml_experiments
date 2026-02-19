# Iris Dataset – Linear Discriminant Analysis (Python and R)

## What is this?
This project contains a pair of implementations (R and python) that perform Linear Discriminant Analysis (LDA) using the famous Iris dataset
The goal is simple: Given measurements of a flower (like petal length and sepal width), can we correctly identify which species it belongs to?

## Why is this interesting?
This project shows how a computer can learn to sort things into categories based on measurable traits.
In everyday terms, this is the same kind of task used in:
- Credit approval systems
- Medical diagnosis support
- Customer segmentation
- Face recognition

## What does this project show?
This project is an example of:
- How computers can sort data into categories
- How numerical measurements can be used to make predictions
- How to train and evaluate a classification model
- Implementing the same analysis in both Python and R


# Technical Stuff

## How do I execute the code?
Depending on which script you choose, you just need to execute it in python or R.
If you execute the python script you need the sklearn library before executing the script.
To get that library type: 
pip3 install sklearn

## How do I interpret the results?
The models present a lot of valuable diagnostics.
Among them, two pieces of information are easily interpretable:

1) The confusion matrix
It should look something like this:

                 Predicted
               Setosa  Versicolor  Virginica
Actual
Setosa            11        0           0
Versicolor         0       12           1
Virginica          0        0          14

2) The plot
The models produce a scatterplot-like plot where colors denote different groups.
A good model shows clearly separated clusters for each class with minimal overlap.


