# ML Experiments

A portfolio of self-contained machine learning experiments covering supervised learning, unsupervised learning, NLP, dimensionality reduction, and LLM integration — implemented in Python and R.

---

## Skills at a glance

| Domain | Techniques |
|---|---|
| Supervised learning | Logistic regression, Naïve Bayes, random forest, binary classification |
| Unsupervised learning | K-means (from scratch), customer segmentation, color quantization |
| Dimensionality reduction | Linear Discriminant Analysis, feature projection |
| NLP & text generation | Markov chains, n-gram language models, text preprocessing |
| LLM integration | OpenAI API, prompt engineering, stateful conversation |
| End-to-end analysis | EDA, feature engineering, model comparison, visualization |
| Stack | Python, R, scikit-learn, pandas, matplotlib, seaborn, scipy |

---

## Projects

| Project | Area | Techniques | Stack |
|---|---|---|---|
| [`bayes_spam_detection`](https://github.com/Federico-CM/ml_experiments/tree/main/projects/bayes_spam_detection) | Supervised learning | Naïve Bayes, probabilistic classification, text preprocessing | Python |
| [`chatbot_implementation_chatGPTAPI`](https://github.com/Federico-CM/ml_experiments/tree/main/projects/chatbot_implementation_chatGPTAPI) | LLM / API integration | OpenAI API, prompt engineering, conversation state management | Python |
| [`full_analisis_ifood`](https://github.com/Federico-CM/ml_experiments/tree/main/projects/full_analisis_ifood) | End-to-end analysis | EDA, feature engineering, logistic regression, random forest | Python |
| [`kmeans_from_scratch_image_simplifier`](https://github.com/Federico-CM/ml_experiments/tree/main/projects/kmeans_from_scratch_image_simplifier) | Unsupervised learning | K-means (from scratch), color quantization, pixel clustering | Python |
| [`kmeans_customer_segmentation`](https://github.com/Federico-CM/ml_experiments/tree/main/projects/kmeans_customer_segmentation) | Unsupervised learning | K-means, customer profiling, elbow method, cluster interpretation | Python |
| [`lda_iris_segmentation`](https://github.com/Federico-CM/ml_experiments/tree/main/projects/lda_iris_segmentation) | Dimensionality reduction | Linear Discriminant Analysis, class separability, feature projection | Python & R |
| [`log_reg_cancer_detection`](https://github.com/Federico-CM/ml_experiments/tree/main/projects/log_reg_cancer_detection) | Supervised learning | Logistic regression, binary classification, medical data, model evaluation | Python |
| [`markov_chain_shakespeare`](https://github.com/Federico-CM/ml_experiments/tree/main/projects/markov_chain_shakespeare) | NLP / text generation | Markov chains, n-gram modeling, probabilistic sampling | Python & R |

---

## Getting started

```bash
# Python
python3 projects/<project_name>/<script>.py

# R
Rscript projects/<project_name>/<script>.R
```

```bash
pip install numpy pandas scikit-learn matplotlib seaborn scipy
```

**Special requirements**

| Project | Requirement |
|---|---|
| `chatbot_implementation_chatGPTAPI` | `OPENAI_API_KEY` set as environment variable |
| `markov_chain_shakespeare` | `shakespeare.txt` in the project directory |
| `kmeans_from_scratch_image_simplifier` | An image file in the project directory |
| `full_analisis_ifood` | Run in order: `process_data.py` → `eda.py` → `logreg.py` / `r_forest.py` |

---

## Structure

```
projects/                   # Individual experiments (each self-contained)
common/data/                # Shared datasets
templates/project_template/ # Boilerplate for new projects
```

Each project has its own README with purpose, approach, and result interpretation.
