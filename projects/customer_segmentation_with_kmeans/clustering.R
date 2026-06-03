# ------------------------------------------------------------
# K-means of synthetic customer data 
# Plotting at each iteration
# ------------------------------------------------------------

set.seed(42)

# 1) Generate synthetic customer data
# not super realistic but easy to understand
n_per_cluster <- 150

# income_k, spend_score
true_centers <- rbind(
  c(50,  35),
  c(75,  55),
  c(105, 60)
)

sd_income <- 10
sd_spend  <- 8

make_cluster <- function(center, n, sd_income, sd_spend) {
  cbind(
    income = rnorm(n, mean = center[1], sd = sd_income),
    spending = rnorm(n, mean = center[2], sd = sd_spend)
  )
}

X <- rbind(
  make_cluster(true_centers[1, ], n_per_cluster, sd_income, sd_spend),
  make_cluster(true_centers[2, ], n_per_cluster, sd_income, sd_spend),
  make_cluster(true_centers[3, ], n_per_cluster, sd_income, sd_spend)
)

df <- data.frame(
  income = X[,1],
  spending = X[,2]
)


# 2) Manual K-means implementation
k <- 3
max_iter <- 30
tolerance <- 1e-4

init_idx <- sample(1:nrow(df), k)
centroids <- as.matrix(df[init_idx, ])

for (iter in 1:max_iter) {
  
  # --- Assignment ---
  # squared euclidean distances
  distances <- sapply(seq_len(k), function(j) {
    income_diff   <- df$income   - centroids[j, 1]
    spending_diff <- df$spending - centroids[j, 2]
    
    income_diff^2 + spending_diff^2
  })
  
  cluster_assign <- apply(distances, 1, which.min)
  
  # --- Plot ---
  plot(
    df$income, df$spending,
    col = cluster_assign,
    pch = 19, cex = 0.8,
    main = paste("Iteration", iter),
    xlab = "Annual Income (k$)",
    ylab = "Spending Score (0-100)",
    xlim = c(15, 140),
    ylim = c(0, 100)
  )
  
  points(
    centroids[,1], centroids[,2],
    pch = 4, cex = 2.5, lwd = 3,
    col = 1:k
  )
  
  Sys.sleep(1)
  
  # --- Update ---
  new_centroids <- matrix(0, nrow = k, ncol = 2)
  
  for (j in 1:k) {
    cluster_points <- df[cluster_assign == j, ]
    new_centroids[j, ] <- colMeans(cluster_points)
  }
  
  shift <- sum((centroids - new_centroids)^2)
  
  if (shift < tolerance) {
    cat("Converged at iteration", iter, "\n")
    break
  }
  
  centroids <- new_centroids
}

cat("\nFinal centroids:\n")
print(centroids)