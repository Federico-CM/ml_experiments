# lda_iris.R

# Load required library
library(MASS)

# Load dataset (built into R)
data(iris)

# Set seed for reproducibility
set.seed(42)

# Train/test split (75% train, 25% test)
n <- nrow(iris)
train_index <- sample(1:n, size = 0.75 * n)

train_data <- iris[train_index, ]
test_data  <- iris[-train_index, ]

# Fit LDA model
lda_model <- lda(Species ~ ., data = train_data)

# Print model summary
print(lda_model)

# Predict on test set
pred <- predict(lda_model, test_data)

# Predicted classes
predicted_classes <- pred$class

# Confusion matrix
conf_matrix <- table(Predicted = predicted_classes,
                     Actual = test_data$Species)
print(conf_matrix)

# Accuracy
accuracy <- mean(predicted_classes == test_data$Species)
cat("\nAccuracy:", accuracy, "\n")

# Optional: Plot LDA projection (first two linear discriminants)
plot(pred$x[, 1], pred$x[, 2],
     col = as.numeric(test_data$Species),
     pch = 19,
     xlab = "LD1",
     ylab = "LD2",
     main = "Iris - LDA Projection")
legend("topright",
       legend = levels(iris$Species),
       col = 1:length(levels(iris$Species)),
       pch = 19)

