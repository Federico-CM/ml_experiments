# log_reg_breast_cancer_like_sklearn.R

suppressPackageStartupMessages({
  library(caret)
  library(glmnet)  # for ridge-penalized logistic regression
})

set.seed(42)

# Load data (make sure wd is correct)
df <- read.csv("breast_cancer_sklearn.csv")

# Train/test split
train_idx <- createDataPartition(df$target, p = 0.75, list = FALSE)
y_col <- which(names(df) == "target")

X_train <- df[train_idx, -y_col]
X_test  <- df[-train_idx, -y_col]
y_train <- df[train_idx,  y_col]
y_test  <- df[-train_idx, y_col]

# Standardize using TRAIN stats
mu  <- sapply(X_train, mean, na.rm = TRUE)
sdv <- sapply(X_train, sd,   na.rm = TRUE)

x_train <- scale(as.matrix(X_train), center = mu, scale = sdv)
x_test  <- scale(as.matrix(X_test),  center = mu, scale = sdv)

# Labels as 0/1 numeric
y_train_num <- as.numeric(y_train)
y_test_num  <- as.numeric(y_test)

# Fit logistic regression
C <- 1.0
n <- nrow(x_train)
lambda_like_sklearn <- 1 / (2 * C * n)

fit <- glmnet(
  x = x_train,
  y = y_train_num,
  family = "binomial",
  alpha = 0,                 # ridge (L2), like sklearn default
  lambda = lambda_like_sklearn,
  standardize = FALSE
)

# Predict probabilities and classes
p_hat  <- as.numeric(predict(fit, newx = x_test, s = lambda_like_sklearn, type = "response"))
y_pred <- ifelse(p_hat >= 0.5, 1L, 0L)

# Evaluate (similar outputs)
acc <- mean(y_pred == y_test_num)
cat("Accuracy:", acc, "\n\n")

y_test_f <- factor(y_test_num, levels = c(0, 1))
y_pred_f <- factor(y_pred,     levels = c(0, 1))

cm <- confusionMatrix(y_pred_f, y_test_f, positive = "1")
print(cm)



