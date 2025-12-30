# Load required libraries
library(corrplot)
library(ggplot2)
library(dplyr)
library(caret)
library(randomForest)

## Load Data
day_data <- read.csv("day.csv", header = TRUE)
hour_data <- read.csv("hour.csv", header = TRUE)

### Data Preparation
cat("Checking for missing values:\n")
print(colSums(is.na(hour_data)))

cat("\nChecking for duplicate rows:\n")
duplicates <- nrow(hour_data[duplicated(hour_data), ])
cat("Number of duplicate rows:", duplicates, "\n")

# Convert categorical columns to factors
hour_data <- hour_data %>%
  mutate(
    season = factor(season, levels = c(1, 2, 3, 4), labels = c("Spring", "Summer", "Fall", "Winter")),
    yr = factor(yr, levels = c(0, 1), labels = c("2011", "2012")),
    holiday = factor(holiday, levels = c(0, 1), labels = c("No", "Yes")),
    workingday = factor(workingday, levels = c(0, 1), labels = c("No", "Yes")),
    weathersit = factor(weathersit, levels = c(1, 2, 3, 4), 
                        labels = c("Clear", "Mist", "Light Snow/Rain", "Heavy Rain"))
  )

# Add rush hour feature
hour_data <- hour_data %>%
  mutate(rush_hour = ifelse((hr >= 7 & hr <= 9) | (hr >= 17 & hr <= 19), 1, 0))

# Display the first few rows of the dataframe to verify the changes
cat("First few rows of the dataframe after adding rush_hour feature:\n")
print(head(hour_data))

#Outlier Detection and Removal
cat("\nRemoving Outliers and Revisualizing:\n")
hour_data_cleaned <- hour_data  # Copy for cleaning

variables <- c("cnt", "temp", "atemp", "hum", "windspeed")
for (var in variables) {
  Q1 <- quantile(hour_data_cleaned[[var]], 0.25, na.rm = TRUE)
  Q3 <- quantile(hour_data_cleaned[[var]], 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR
  upper_bound <- Q3 + 1.5 * IQR
  
  outliers <- sum(hour_data_cleaned[[var]] < lower_bound | hour_data_cleaned[[var]] > upper_bound, na.rm = TRUE)
  cat("\nVariable:", var, "Number of Outliers:", outliers, "\n")
  
  # Remove outliers
  hour_data_cleaned <- hour_data_cleaned %>%
    filter(hour_data_cleaned[[var]] >= lower_bound & hour_data_cleaned[[var]] <= upper_bound)
}

cat("\nDataset Dimensions After Outlier Removal:\n")
print(dim(hour_data_cleaned))

## Measures of Central Tendency
cat("\nMeasures of Central Tendency:\n")
for (var in variables) {
  cat("\nVariable:", var, "\n")
  cat("Mean:", mean(hour_data_cleaned[[var]], na.rm = TRUE), "\n")
  cat("Median:", median(hour_data_cleaned[[var]], na.rm = TRUE), "\n")
  cat("Mode:", names(sort(table(hour_data_cleaned[[var]]), decreasing = TRUE))[1], "\n")
}
## EDA 
# Correlation Matrix (with numeric columns only)
numeric_cols <- hour_data %>% select(where(is.numeric))
correlation_matrix <- cor(numeric_cols, use = "complete.obs")
corrplot(correlation_matrix, method = "color", type = "upper", tl.col = "black", tl.srt = 45, 
         addCoef.col = "black", title = "Annotated Correlation Matrix")
# Count of Bikes During Different Months
ggplot(hour_data, aes(x = factor(mnth), y = cnt, fill = season)) +
  geom_boxplot() +
  labs(title = "Count of Bikes During Different Months", x = "Month", y = "Rental Count") +
  theme_minimal()

# Count of Bikes During Different Hours
ggplot(hour_data, aes(x = factor(hr), y = cnt, fill = workingday)) +
  geom_boxplot() +
  labs(title = "Count of Bikes During Different Hours", x = "Hour", y = "Rental Count") +
  theme_minimal()

# Relationship Between Windspeed and Bike Rentals
ggplot(hour_data, aes(x = windspeed, y = cnt)) +
  geom_point(alpha = 0.5, color = "blue") +
  geom_smooth(method = "lm", color = "red", se = FALSE) +
  labs(title = "Relationship Between Windspeed and Bike Rentals", x = "Windspeed", y = "Rental Count") +
  theme_minimal()

# Relationship Between Humidity and Bike Rentals
ggplot(hour_data, aes(x = hum, y = cnt)) +
  geom_point(alpha = 0.5, color = "green") +
  geom_smooth(method = "lm", color = "red", se = FALSE) +
  labs(title = "Relationship Between Humidity and Bike Rentals", x = "Humidity", y = "Rental Count") +
  theme_minimal()

# Relationship Between Feeling Temperature and Bike Rentals
ggplot(hour_data, aes(x = atemp, y = cnt)) +
  geom_point(alpha = 0.5, color = "purple") +
  geom_smooth(method = "lm", color = "red", se = FALSE) +
  labs(title = "Relationship Between Feeling Temperature and Bike Rentals", x = "Feeling Temperature (atemp)", y = "Rental Count") +
  theme_minimal()

# Relationship Between Rush Hour and Bike Rentals
ggplot(hour_data, aes(x = factor(rush_hour), y = cnt, fill = factor(rush_hour))) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Impact of Rush Hour on Bike Rentals", x = "Rush Hour (0 = Non-Rush, 1 = Rush)", y = "Rental Count") +
  scale_fill_manual(values = c("0" = "blue", "1" = "orange"), labels = c("Non-Rush Hour", "Rush Hour")) +
  theme_minimal()

##Model Preperation

# One-Hot Encoding
categorical_vars <- c("season", "yr", "holiday", "workingday", "weathersit")
encoded_data <- model.matrix(~ . - 1, data = hour_data[, categorical_vars]) %>% as.data.frame()
hour_data <- cbind(hour_data %>% select(-all_of(categorical_vars)), encoded_data)

# Feature Importance Using Random Forest
set.seed(123)
X <- hour_data %>% select(-cnt, -casual, -registered, -instant, -dteday)  # Exclude target and irrelevant columns
Y <- hour_data$cnt  # Target variable
rf_model <- randomForest(x = X, y = Y, ntree = 100, importance = TRUE)
feature_importance <- importance(rf_model)
important_features <- names(sort(feature_importance[, "IncNodePurity"], decreasing = TRUE))[1:10]
cat("\nTop 10 Features Based on Importance:\n")
print(important_features)
X <- X[, important_features]

# Train-Test Split
set.seed(123)
trainIndex <- createDataPartition(Y, p = 0.8, list = FALSE)
x_train <- X[trainIndex, ]
x_test <- X[-trainIndex, ]
y_train <- Y[trainIndex]
y_test <- Y[-trainIndex]

# Baseline Model Comparison
baseline_pred <- mean(y_train)  # Predict the mean of the training data
baseline_mse <- mean((y_test - baseline_pred)^2)
baseline_rmse <- sqrt(baseline_mse)
baseline_r_squared <- 1 - sum((y_test - baseline_pred)^2) / sum((y_test - mean(y_test))^2)
cat("\nBaseline Model Metrics:\n")
cat("Baseline Mean Squared Error (MSE):", baseline_mse, "\n")
cat("Baseline Root Mean Squared Error (RMSE):", baseline_rmse, "\n")
cat("Baseline R-Squared Value:", baseline_r_squared, "\n")

# Random Forest Model
rf_model <- randomForest(x = x_train, y = y_train, ntree = 100, importance = TRUE)
rf_predictions <- predict(rf_model, newdata = x_test)

# Evaluation Metrics for Random Forest
rf_mse <- mean((y_test - rf_predictions)^2)
rf_rmse <- sqrt(rf_mse)
rf_mae <- mean(abs(y_test - rf_predictions))
rf_r_squared <- 1 - sum((y_test - rf_predictions)^2) / sum((y_test - mean(y_test))^2)

cat("\nRandom Forest Model Metrics:\n")
cat("Random Forest Mean Squared Error (MSE):", rf_mse, "\n")
cat("Random Forest Root Mean Squared Error (RMSE):", rf_rmse, "\n")
cat("Random Forest Mean Absolute Error (MAE):", rf_mae, "\n")
cat("Random Forest R-Squared Value:", rf_r_squared, "\n")

# Visualization
comparison_data <- data.frame(Actual = y_test, Predicted = rf_predictions)
ggplot(comparison_data, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.5, color = "blue") +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  labs(title = "Actual vs Predicted Bike Rentals (Random Forest)", x = "Actual Count", y = "Predicted Count") +
  theme_minimal()
