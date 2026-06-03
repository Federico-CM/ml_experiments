# Model Comparison Report

## Summary

Two classification models were evaluated to predict customer response to the marketing campaign using the cleaned and processed iFood dataset:

- Logistic Regression
- Random Forest

Both models achieved good overall performance, with Logistic Regression slightly outperforming Random Forest across most evaluation metrics.

## Impact
**The Logistic Regression model demonstrates meaningful targeting capability.** While responders represent approximately 15% of the customer population (66 of 447 customers), the model identifies a smaller target segment comprising approximately 9% of customers (41 of 447) in which 71% are actual responders. The overal recall of the model is 44%.

In practical terms, **customers selected by the model are nearly five times more likely to respond than a randomly selected customer from the overall population.** This indicates that the model effectively concentrates likely responders into a much smaller group while maintaining a relatively high precision rate.

These results suggest that the model could improve marketing targeting efficiency by directing campaign efforts toward customers with a substantially higher probability of responding.

The recall rate of 44% indicates that the model identifies fewer than half of all actual responders, meaning that a targeted campaign based solely on model predictions would miss a considerable proportion of potential respondents. This highlights that **there is a trade-off between improved targeting efficiency and overall response volume, as gains in precision come at the cost of excluding some likely responders.**

## Model Performance

| Metric | Logistic Regression | Random Forest |
|----------|----------:|----------:|
| Accuracy | 89.0% | 87.9% |
| Precision (Response = 1) | 71% | 70% |
| Recall (Response = 1) | 44% | 32% |
| F1-score (Response = 1) | 0.54 | 0.44 |
| ROC AUC | 0.894 | 0.877 |
| PR AUC | 0.616 | 0.588 |

Logistic Regression achieved better recall, F1-score, ROC AUC, and PR AUC, indicating stronger overall predictive performance.

---

## Confusion Matrices

### Logistic Regression

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 369 | 12 |
| Actual 1 | 37 | 29 |

### Random Forest

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 372 | 9 |
| Actual 1 | 45 | 21 |

Both models correctly classified most non-responders, but Logistic Regression identified more actual responders.

---

## Important Predictors

Both models identified similar factors as important for campaign response:

- Customer tenure
- Previous campaign acceptance
- Customer spending patterns
- Income
- Purchase behavior
- Recency

Customer tenure and recency were consistently among the strongest predictors.

---

## Conclusion

Both models demonstrated strong performance in predicting campaign response, with Logistic Regression outperforming Random Forest across key evaluation metrics. The Logistic Regression model effectively identified likely responders, increasing response concentration from 15% in the overall customer base to 71% within the targeted segment. Customer tenure, recency, prior campaign acceptance, and spending behavior were the most influential predictors. Overall, the results indicate that Logistic Regression is a suitable model for improving marketing campaign targeting and efficiency. If the objective is maximizing response rate while limiting campaign volume, Logistic Regression should be used to prioritize customers.
