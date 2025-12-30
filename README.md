# Bike-Rental-Demand-Prediction

This project analyzes bike-sharing usage patterns and applies supervised machine learning techniques to predict bike rental demand based on temporal and environmental factors. The objective is to identify key demand drivers and evaluate model performance using both baseline and non-linear regression approaches.

---

## Data Preparation & Feature Engineering

- Verified data quality by checking for missing values and duplicates  
- Engineered a **rush_hour** feature to capture peak commuting periods  
- Removed extreme outliers from key numerical variables using the IQR method  
- Applied one-hot encoding for categorical features prior to modeling  

---

## Exploratory Insights

Exploratory analysis revealed clear and consistent demand patterns:
- Bike usage is significantly higher during **rush-hour periods**
- Demand varies systematically across seasons and months
- Weather conditions, particularly temperature, strongly influence rental volume

The following visualization highlights the impact of rush hours on bike rental demand and motivates the inclusion of temporal features in the model.

### Impact of Rush Hour on Bike Rentals
![Impact of Rush Hour on Bike Rentals](images/rush_hour_impact.png)

---

## Modeling Approach

Two regression-based models were developed and compared:

### Baseline Model — Linear Regression
- Served as an interpretable benchmark
- Explained approximately **56% of the variance** in bike rental demand

### Advanced Model — Random Forest Regression
- Captured non-linear relationships between temporal and environmental variables
- Significantly outperformed the baseline model
- Achieved an **R² of approximately 0.93**, with substantially lower prediction error

Model performance was evaluated using:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared (R²)

---

## Model Performance

The following plot compares actual bike rental counts with predictions from the Random Forest model. Each point represents a single observation, and the diagonal line indicates perfect prediction. The close alignment of points to this reference line demonstrates strong predictive performance across low and high demand levels.

### Actual vs Predicted Bike Rentals (Random Forest)
![Actual vs Predicted Bike Rentals (Random Forest)](images/actual_vs_predicted_rf.png)

---

## Key Takeaways

- **Rush-hour patterns** are a primary driver of bike rental demand  
- **Temperature and time-related features** play a significant role in usage behavior  
- **Random Forest regression** substantially improves demand estimation over linear models  
- Combining domain-aware feature engineering with non-linear models yields strong performance  

---

## Note

This project was developed as part of my Master of Data Science program, within the
Advanced Applied Statistics course.

---

## Contact
For any questions, please contact me:

- [LinkedIn](https://www.linkedin.com/in/mashael-alsogair-97b754230/)

Thank you!
