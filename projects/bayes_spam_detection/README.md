# Naive Bayes Spam Detector

## What is this?
This project contains a Python implementation of a Naive Bayes spam detector built from scratch.

The goal is simple: Given a text message, can a computer determine whether it is spam or not based on the words it contains?

The model learns from a labeled dataset of SMS messages and calculates the probability that a new message belongs to the "spam" or "ham" (not spam) category. After training, users can enter messages interactively and receive a prediction in real time.

---

## Why is this interesting?
This project demonstrates one of the most widely used techniques in machine learning and natural language processing.

In everyday terms, similar approaches are used in:

- Email spam filtering
- SMS spam detection
- Customer support ticket classification
- Sentiment analysis
- Document categorization
- Content moderation systems

Even though the underlying mathematics is relatively simple, Naive Bayes classifiers often perform surprisingly well on text classification problems.

---

## What does this project show?
This project is an example of:

- How a machine can learn patterns from labeled data
- How probabilities can be used for classification
- How text messages can be converted into numerical information
- How word frequencies influence predictions
- How Laplace smoothing handles previously unseen words
- How a classic machine learning algorithm works under the hood

---

# Technical Stuff

## How does the algorithm work?

The classifier follows the Naive Bayes approach:

1. Read a dataset containing spam and non-spam messages.
2. Break each message into individual words (tokenization).
3. Count how often each word appears in spam messages and non-spam messages.
4. Calculate prior probabilities for both classes.
5. Use Bayes' Theorem to estimate the probability that a new message belongs to each class.
6. Choose the class with the higher probability.

To avoid numerical underflow caused by multiplying many small probabilities together, the implementation uses logarithms of probabilities instead.

The model also uses Laplace smoothing, which prevents words that never appeared during training from forcing probabilities to zero.

---

## How do I execute the code?

Make sure the dataset file (`SMSSpamCollection.csv`) is in the same directory as the Python script before running it.

Execute the program using:

```bash
python3 spam_detector.py
