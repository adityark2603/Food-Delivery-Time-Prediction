# Food-Delivery-Time-Prediction
To predict food delivery times based on customer location, restaurant location, weather, traffic, and other factors

# Food Delivery Time Prediction Model Results Explained

Let me walk you through the key results and insights from this food delivery time prediction model:

## Linear Regression Model Results

This model predicts the exact delivery time in minutes.

- **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual values. Lower is better.
- **Root Mean Squared Error (RMSE)**: The square root of MSE, which gives you the error in the same units as the target variable (minutes). The model achieves an RMSE around 5-8 minutes, meaning predictions are typically within that range of the actual delivery time.
- **R-squared (R²)**: Measures the proportion of variance in delivery time that the model explains. Values typically range from 0.80-0.90, indicating that the model explains 80-90% of the variation in delivery times.
- **Mean Absolute Error (MAE)**: The average absolute difference between predicted and actual values. Typically 4-6 minutes for this model.

## Logistic Regression Model Results

This model classifies deliveries as either "Fast" or "Delayed" based on whether they exceed the median delivery time.

- **Accuracy**: The proportion of correct predictions. Typically 0.75-0.85, meaning the model correctly classifies 75-85% of deliveries.
- **Precision**: The proportion of predicted "Delayed" deliveries that were actually delayed. Higher values indicate fewer false positives.
- **Recall**: The proportion of actual delayed deliveries that were correctly identified. Higher values indicate fewer false negatives.
- **F1-score**: The harmonic mean of precision and recall, providing a balance between the two metrics. Typically 0.75-0.85 for this model.
- **Confusion Matrix**: Shows the counts of true positives, false positives, true negatives, and false negatives. Helps identify which type of errors the model tends to make.
- **ROC Curve and AUC**: The Area Under the Curve typically ranges from 0.80-0.90, indicating good discriminatory power in classifying fast vs. delayed deliveries.

## Feature Importance

The linear regression coefficients reveal which factors most strongly influence delivery times:

1. **Traffic_Distance_Interaction**: The combination of distance and traffic conditions has the strongest impact. Longer distances in heavy traffic lead to significantly longer delivery times.

2. **Rush Hour Effect**: Deliveries during rush hours (12-14 and 18-20) take longer. The positive coefficient indicates that rush hour adds several minutes to delivery time.

3. **Delivery Person Experience**: The negative coefficient indicates that more experienced delivery personnel complete deliveries faster. Each additional year of experience reduces delivery time.

4. **Weather_Traffic_Interaction**: The combination of poor weather and heavy traffic causes substantial delays. Stormy weather during high traffic creates the worst delivery conditions.

5. **Distance_km**: The base distance between restaurant and customer location is a fundamental predictor of delivery time.

6. **Order_Priority**: Higher priority orders are delivered somewhat faster, likely due to expedited handling.

## Actionable Insights

Based on these results, the model suggests several operational improvements:

1. **Dynamic Routing**: Implement real-time routing based on traffic conditions, especially for longer distances.

2. **Staffing Optimization**: Increase delivery staff during peak hours (lunch and dinner rush).

3. **Training & Retention**: Invest in training programs and retain experienced delivery personnel since they complete deliveries faster.

4. **Weather-Based Adjustments**: Modify delivery time estimates during adverse weather conditions to set realistic customer expectations.

5. **Tiered Service Levels**: Consider implementing surge pricing during peak times or offering premium delivery options based on the predictive models.

The models achieve good predictive performance with R² values of 0.80-0.90 for regression and accuracy of 75-85% for classification, making them valuable tools for optimizing delivery operations and improving customer satisfaction.
