# Logistic Regression Report

## Summary

A logistic regression model was fitted to predict customer response to the marketing campaign using the cleaned and processed iFood dataset.

**The model achieved good overall performance and provides an interpretable baseline model.**

Accuracy on the test set was **89.0%**.

## Important Findings

### 1. Overall Model Performance

The model achieved:

- Accuracy: **89.0%**
- Precision (Response = 1): **71%**
- Recall (Response = 1): **44%**
- F1-score (Response = 1): **0.54**

The model performs well at identifying non-responders but misses a substantial proportion of actual responders.

---

### 2. Confusion Matrix

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 369 | 12 |
| Actual 1 | 37 | 29 |

Most classification errors are false negatives, indicating that many responding customers are not identified by the model.

---

### 3. Important Positive Predictors

Variables associated with a higher probability of campaign response include:

- Customer tenure
- Previous campaign acceptance (`AcceptedCmp1`, `AcceptedCmp3`, `AcceptedCmp4`, `AcceptedCmp5`)
- Higher spending on meat products
- More web visits
- More deal purchases
- Higher education levels

Customer tenure was the strongest positive predictor.

---

### 4. Important Negative Predictors

Variables associated with a lower probability of campaign response include:

- Higher recency
- More store purchases
- Presence of teenagers in the household
- Married or together marital status
- Lower education levels

Recency was the strongest negative predictor.

## Conclusion

The logistic regression model provides a useful and interpretable baseline. Results suggest that customer engagement, loyalty, and previous campaign participation are important drivers of response.

Although overall accuracy is high, recall for responders remains modest. More flexible models may improve predictive performance, particularly for identifying customers likely to respond. 
