# Predicting House Prices Using Machine Learning

## Project Overview
This project uses a machine learning model to predict house prices based on various features of the properties. The model is trained using a dataset with features such as the number of bedrooms, bathrooms, square footage, and more. The final model is built using a **RandomForestRegressor** and aims to minimize the prediction error for house prices.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Model Building](#model-building)
- [Model Evaluation](#model-evaluation)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [License](#license)
- [How to Contribute](#how-to-contribute)

## Dataset
The dataset contains various features like:
- **bedrooms**: Number of bedrooms
- **bathrooms**: Number of bathrooms
- **sqft_living**: Square footage of the living area
- **floors**: Number of floors
- And other additional categorical and numeric features.

**Target Variable:**
- **price**: The actual house price.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/house-price-prediction.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Ensure the following libraries are installed:
    - `pandas`
    - `numpy`
    - `scikit-learn`
    - `matplotlib` (optional, for data visualization)

## Data Preprocessing
1. **Load the Dataset:**
    ```python
    import pandas as pd
    data = pd.read_csv('data.csv')
    ```
2. **Handle Missing Values:**
    ```python
    data.fillna(0, inplace=True)
    ```
3. **Feature Encoding:**
    ```python
    data = pd.get_dummies(data, drop_first=True)
    ```
4. **Split the Data:**
    ```python
    X = data.drop(columns=['price'])
    y = data['price']
    ```
5. **Train-Test Split:**
    ```python
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```

## Model Building
1. **Random Forest Regressor:**
    ```python
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    ```
2. **Make Predictions:**
    ```python
    predictions = model.predict(X_test)
    ```

## Model Evaluation
1. **Evaluate the Model Performance:**
    ```python
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    print(f'MAE: {mae}')
    print(f'RMSE: {rmse}')
    ```
2. **Compare Actual vs Predicted:**
    ```python
    result = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
    print(result.head())
    ```

## Conclusion
- The **RandomForestRegressor** model is used to predict house prices based on the input features.
- The model performs well for mid-range properties but struggles with very high or low price ranges.
- Future improvements can include further feature engineering, data cleaning, and testing other models to improve prediction accuracy.



