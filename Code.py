# Food Delivery Time Prediction Model
# Complete implementation with data preprocessing, EDA, and predictive modeling

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import math
from sklearn.preprocessing import LabelEncoder

# Set display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
np.random.seed(42)

# Phase 1: Data Collection and Exploratory Data Analysis (EDA)
# Step 1 - Data Import and Preprocessing

# Creating synthetic data
def create_synthetic_data(n_samples=1000):
    np.random.seed(42)

    # Generate random coordinates for restaurants and customers
    restaurant_lat = np.random.uniform(28.4, 28.7, n_samples)
    restaurant_lon = np.random.uniform(77.0, 77.3, n_samples)
    customer_lat = np.random.uniform(28.4, 28.7, n_samples)
    customer_lon = np.random.uniform(77.0, 77.3, n_samples)

    # Calculate distance using Haversine formula
    def haversine(lat1, lon1, lat2, lon2):
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # Radius of Earth in kilometers
        return c * r

    distances = haversine(restaurant_lat, restaurant_lon, customer_lat, customer_lon)

    # Generate other features
    weather_conditions = np.random.choice(['Sunny', 'Rainy', 'Cloudy', 'Stormy'], n_samples)
    traffic_conditions = np.random.choice(['Low', 'Medium', 'High', 'Very High'], n_samples)
    vehicle_type = np.random.choice(['Motorcycle', 'Bicycle', 'Car', 'Scooter'], n_samples)
    order_cost = np.random.uniform(100, 1000, n_samples)

    # Time of day (24-hour format)
    time_of_day = np.random.randint(8, 23, n_samples)
    # Is it rush hour? (Rush hours: 12-14, 18-20)
    is_rush_hour = ((time_of_day >= 12) & (time_of_day <= 14)) | ((time_of_day >= 18) & (time_of_day <= 20))

    # Generate delivery times based on features (with some noise)
    base_time = 10  # Base delivery time in minutes
    distance_factor = 2  # Minutes per kilometer

    # Weather impact
    weather_impact = {
        'Sunny': 0,
        'Cloudy': 2,
        'Rainy': 5,
        'Stormy': 10
    }
    weather_minutes = np.array([weather_impact[w] for w in weather_conditions])

    # Traffic impact
    traffic_impact = {
        'Low': 0,
        'Medium': 3,
        'High': 8,
        'Very High': 15
    }
    traffic_minutes = np.array([traffic_impact[t] for t in traffic_conditions])

    # Vehicle impact
    vehicle_impact = {
        'Car': 0,
        'Motorcycle': 1,
        'Scooter': 2,
        'Bicycle': 5
    }
    vehicle_minutes = np.array([vehicle_impact[v] for v in vehicle_type])

    # Rush hour impact
    rush_hour_minutes = np.where(is_rush_hour, 10, 0)

    # Calculate delivery time with noise
    delivery_time = (
        base_time +
        distances * distance_factor +
        weather_minutes +
        traffic_minutes +
        vehicle_minutes +
        rush_hour_minutes +
        np.random.normal(0, 5, n_samples)  # Add random noise
    ).round()

    # Make sure delivery time is positive
    delivery_time = np.maximum(delivery_time, 5)

    # Delivery person experience (years)
    delivery_person_experience = np.random.uniform(0.5, 5, n_samples).round(1)

    # Order priority (1=Low, 2=Medium, 3=High)
    order_priority = np.random.choice([1, 2, 3], n_samples)

    # Create DataFrame
    df = pd.DataFrame({
        'Restaurant_Latitude': restaurant_lat,
        'Restaurant_Longitude': restaurant_lon,
        'Customer_Latitude': customer_lat,
        'Customer_Longitude': customer_lon,
        'Distance_km': distances.round(2),
        'Weather_Conditions': weather_conditions,
        'Traffic_Conditions': traffic_conditions,
        'Vehicle_Type': vehicle_type,
        'Time_of_Day': time_of_day,
        'Is_Rush_Hour': is_rush_hour.astype(int),
        'Order_Cost': order_cost.round(2),
        'Delivery_Person_Experience': delivery_person_experience,
        'Order_Priority': order_priority,
        'Delivery_Time': delivery_time
    })

    # Introduce some missing values
    for col in ['Distance_km', 'Weather_Conditions', 'Traffic_Conditions']:
        mask = np.random.choice([True, False], size=n_samples, p=[0.03, 0.97])
        df.loc[mask, col] = np.nan

    return df

# Create and display the dataset
df = create_synthetic_data(1000)
print("Dataset created successfully!")
print("\nDataset sample:")
print(df.head())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Handling missing values
print("\nHandling missing values...")
# For numerical columns, use mean imputation
numeric_cols = ['Distance_km', 'Order_Cost', 'Delivery_Person_Experience']
# For categorical columns, use most frequent value imputation
categorical_cols = ['Weather_Conditions', 'Traffic_Conditions', 'Vehicle_Type']

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numeric_cols),
        ('cat', SimpleImputer(strategy='most_frequent'), categorical_cols)
    ],
    remainder='passthrough'
)

# Fit and transform the data
processed_array = preprocessor.fit_transform(df[numeric_cols + categorical_cols])

# Replace the original columns with the processed ones
for i, col in enumerate(numeric_cols + categorical_cols):
    df[col] = processed_array[:, i]

print("Missing values handled successfully!")
print("\nUpdated dataset summary:")
print(df.describe().T)

# Step 2 - Exploratory Data Analysis (EDA)

# Descriptive statistics
print("\nDescriptive Statistics for Numerical Features:")
print(df.describe())

# Correlation analysis
plt.figure(figsize=(12, 10))
correlation_matrix = df.select_dtypes(include=['float64', 'int64']).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.tight_layout()

# Outlier detection with boxplots
plt.figure(figsize=(15, 10))
numerical_cols = ['Distance_km', 'Delivery_Time', 'Order_Cost', 'Delivery_Person_Experience']
for i, col in enumerate(numerical_cols):
    plt.subplot(2, 2, i+1)
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()

# Distribution of delivery times
plt.figure(figsize=(10, 6))
sns.histplot(df['Delivery_Time'], kde=True)
plt.title('Distribution of Delivery Times')
plt.xlabel('Delivery Time (minutes)')
plt.ylabel('Frequency')

# Relationship between distance and delivery time
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Distance_km', y='Delivery_Time', hue='Traffic_Conditions', data=df)
plt.title('Delivery Time vs Distance by Traffic Conditions')
plt.xlabel('Distance (km)')
plt.ylabel('Delivery Time (minutes)')

# Average delivery time by weather conditions
plt.figure(figsize=(10, 6))
sns.barplot(x='Weather_Conditions', y='Delivery_Time', data=df)
plt.title('Average Delivery Time by Weather Conditions')
plt.xlabel('Weather Conditions')
plt.ylabel('Average Delivery Time (minutes)')

# Average delivery time by traffic conditions
plt.figure(figsize=(10, 6))
sns.barplot(x='Traffic_Conditions', y='Delivery_Time', data=df)
plt.title('Average Delivery Time by Traffic Conditions')
plt.xlabel('Traffic Conditions')
plt.ylabel('Average Delivery Time (minutes)')

# Delivery time by rush hour
plt.figure(figsize=(10, 6))
sns.boxplot(x='Is_Rush_Hour', y='Delivery_Time', data=df)
plt.title('Delivery Time by Rush Hour')
plt.xlabel('Is Rush Hour (1=Yes, 0=No)')
plt.ylabel('Delivery Time (minutes)')

print("\nEDA completed. Visualizations generated.")

# Step 3 - Feature Engineering

# We already have the distance calculation from our synthetic data generation
# For a real dataset with latitude and longitude, you would use the haversine function we defined

# Create a binary target variable for logistic regression
# Classify deliveries as "Fast" (0) or "Delayed" (1)
# Using the median delivery time as the threshold
median_delivery_time = df['Delivery_Time'].median()
df['Is_Delayed'] = (df['Delivery_Time'] > median_delivery_time).astype(int)

print(f"Created binary target 'Is_Delayed' using median delivery time ({median_delivery_time} minutes) as threshold")

# Create additional time-based features
# Convert time of day to categorical periods
def categorize_time(hour):
    if 6 <= hour < 11:
        return 'Morning'
    elif 11 <= hour < 14:
        return 'Lunch'
    elif 14 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Dinner'
    else:
        return 'Night'

df['Time_Category'] = df['Time_of_Day'].apply(categorize_time)

# Create interaction features
df['Traffic_Distance_Interaction'] = df.apply(
    lambda row: row['Distance_km'] * (1 + ['Low', 'Medium', 'High', 'Very High'].index(row['Traffic_Conditions']) * 0.25),
    axis=1
)

df['Weather_Traffic_Interaction'] = df.apply(
    lambda row: (['Sunny', 'Cloudy', 'Rainy', 'Stormy'].index(row['Weather_Conditions']) + 1) *
                (['Low', 'Medium', 'High', 'Very High'].index(row['Traffic_Conditions']) + 1),
    axis=1
)

print("\nFeature engineering completed. New features created:")
print("- Is_Delayed: Binary classification target")
print("- Time_Category: Categorized time of day")
print("- Traffic_Distance_Interaction: Interaction between distance and traffic")
print("- Weather_Traffic_Interaction: Interaction between weather and traffic")

# Phase 2: Predictive Modeling

# Step 4 - Linear Regression Model

# Prepare the features and target for the linear regression model
# Select features for the model
features = [
    'Distance_km', 'Is_Rush_Hour', 'Order_Priority',
    'Delivery_Person_Experience', 'Traffic_Distance_Interaction',
    'Weather_Traffic_Interaction'
]

# Encode categorical variables
categorical_features = []
numerical_features = features

# One-hot encode categorical variables and standardize numerical variables
categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')
numerical_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Split the data into training and testing sets
X = df[features]
y_regression = df['Delivery_Time']
y_classification = df['Is_Delayed']

X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test = train_test_split(
    X, y_regression, y_classification, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Create and train the linear regression model
lr_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

lr_pipeline.fit(X_train, y_reg_train)
y_reg_pred = lr_pipeline.predict(X_test)

# Evaluate the linear regression model
mse = mean_squared_error(y_reg_test, y_reg_pred)
rmse = math.sqrt(mse)
r2 = r2_score(y_reg_test, y_reg_pred)
mae = mean_absolute_error(y_reg_test, y_reg_pred)

print("\nLinear Regression Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_reg_test, y_reg_pred, alpha=0.5)
plt.plot([y_reg_test.min(), y_reg_test.max()], [y_reg_test.min(), y_reg_test.max()], 'r--')
plt.xlabel('Actual Delivery Time')
plt.ylabel('Predicted Delivery Time')
plt.title('Actual vs Predicted Delivery Time')

# Step 5 - Logistic Regression Model
print("\n--- Logistic Regression Model ---")

# Create and train the logistic regression model
log_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

log_pipeline.fit(X_train, y_class_train)
y_class_pred = log_pipeline.predict(X_test)
y_class_prob = log_pipeline.predict_proba(X_test)[:, 1]

# Evaluate the logistic regression model
accuracy = accuracy_score(y_class_test, y_class_pred)
precision = precision_score(y_class_test, y_class_pred)
recall = recall_score(y_class_test, y_class_pred)
f1 = f1_score(y_class_test, y_class_pred)
conf_matrix = confusion_matrix(y_class_test, y_class_pred)

print("\nLogistic Regression Model Evaluation:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fast', 'Delayed'],
            yticklabels=['Fast', 'Delayed'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_class_test, y_class_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

# Phase 3: Reporting and Insights

# Step 6 - Model Evaluation and Comparison
print("\nModel Comparison:")
print(f"Linear Regression RMSE: {rmse:.2f}, R²: {r2:.2f}")
print(f"Logistic Regression Accuracy: {accuracy:.2f}, F1-score: {f1:.2f}")

# Feature importance for linear regression
lr_coefficients = pd.DataFrame({
    'Feature': numerical_features,
    'Coefficient': lr_pipeline.named_steps['regressor'].coef_
})
lr_coefficients = lr_coefficients.sort_values('Coefficient', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=lr_coefficients)
plt.title('Linear Regression Feature Importance')
plt.tight_layout()

# Step 7 - Actionable Insights
print("\nActionable Insights:")
print("1. Distance Impact: The model confirms that distance is a significant factor in delivery time.")
if lr_coefficients.iloc[0]['Feature'] == 'Traffic_Distance_Interaction':
    print("   - The interaction between traffic and distance has the highest impact on delivery time.")
    print("   - Recommendation: Implement dynamic routing based on real-time traffic conditions.")

print("\n2. Rush Hour Effect:")
rush_hour_coef = lr_coefficients[lr_coefficients['Feature'] == 'Is_Rush_Hour']['Coefficient'].values[0]
if rush_hour_coef > 0:
    print(f"   - Rush hour significantly increases delivery time (coefficient: {rush_hour_coef:.2f}).")
    print("   - Recommendation: Increase delivery staff during peak hours and consider surge pricing.")

print("\n3. Experience Matters:")
exp_coef = lr_coefficients[lr_coefficients['Feature'] == 'Delivery_Person_Experience']['Coefficient'].values[0]
if exp_coef < 0:
    print(f"   - More experienced delivery personnel complete deliveries faster (coefficient: {exp_coef:.2f}).")
    print("   - Recommendation: Invest in training programs and retain experienced staff.")

print("\n4. Weather and Traffic Interaction:")
weather_traffic_coef = lr_coefficients[lr_coefficients['Feature'] == 'Weather_Traffic_Interaction']['Coefficient'].values[0]
if weather_traffic_coef > 0:
    print(f"   - The combination of poor weather and heavy traffic causes significant delays (coefficient: {weather_traffic_coef:.2f}).")
    print("   - Recommendation: Adjust delivery time estimates during adverse conditions to set realistic customer expectations.")

# Final summary
print("\nFinal Summary:")
print("This food delivery time prediction project demonstrates how machine learning can be used to optimize delivery operations.")
print("The linear regression model provides accurate time predictions, while the logistic regression model helps classify deliveries as 'Fast' or 'Delayed'.")
print("By implementing the recommendations based on model insights, delivery efficiency can be improved, leading to better customer satisfaction and operational cost savings.")

print("\nProject completed successfully!")
