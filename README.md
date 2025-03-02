# Time Series Sales Forecasting

This repository implements three distinct approaches to forecast sales using the "future_sales" dataset. Each approach processes the data, builds a forecasting model, and evaluates the predictions using RMSE and visualizations. The three approaches are:

1. **LSTM Approach** – Uses a deep learning model based on LSTM networks.
2. **ARIMA Approach** – Uses classical time series modeling with ARIMA.
3. **Hybrid Approach** – Combines a deep learning encoder-decoder with a MLP for improved predictions.

---

## Directory Structure

```
time_series_sales_forecasting/
├── lstm_approach.py         # LSTM approach
├── arima_approach.ipynb     # ARIMA approach
└── hybrid_approach.py       # Hybrid approach 
```

---

## Dataset Description

The dataset consists of:
- **sales_train.csv**: Historical daily sales data.
- **test.csv**: Test set with shop and item IDs.
- **items.csv**: Item information.
- **shops.csv**: Shop details.
- **item_categories.csv**: Item categories.

All CSV files are preprocessed to aggregate daily sales into monthly sales. This aggregated data is then used to create time series for each shop item combination.

---

## Approaches 

### 1. LSTM Approach

**Process:**
- **Data Preparation:**  
  - Aggregates daily sales into monthly figures.
  - Create time series for each shop item combination.
  - Reshapes the series into a format compatible with LSTM networks.
- **Model Building:**  
  - Uses Bidirectional LSTM layers combined with Dropout for regularization and Dense layers for prediction.
  - Early stopping during training.
- **Evaluation:**  
  - Calculates RMSE on both training and validation sets.
  - Visualizes loss curves and prediction vs. actual results.

- Well suited for larger datasets where deep learning can extract features.

---

### 2. ARIMA Approach

**Process:**
- **Stationarity & Transformation:**  
  - Applies the ADF test to assess stationarity.
  - Uses a Box-Cox transformation to stabilize variance.
- **Order Selection:**  
  - Automatically selects ARIMA orders using auto_arima.
  - Provides an option for manual order selection by ACF/PACF plots.
- **Model Fitting & Forecasting:**  
  - Fits an ARIMA model to the transformed data.
  - Forecasts future sales and inverts the transformation.
  - Evaluates performance using RMSE and visualizes forecasts.

- Offers interpretability and simplicity for stationary time series.

---

### 3. Hybrid Approach

**Process:**
- **Data Preparation:**  
  - Data is aggregated monthly and reshaped.
- **Encoder-Decoder Network:**  
  - Trains an LSTM encoder-decoder to learn features from the time series.
- **Hybrid Feature Fusion:**  
  - Encoded features are concatenated with the last month’s sales data to form a hybrid feature set.
- **MLP Regression:**  
  - An MLP model is then trained on these hybrid features to predict future sales.
- **Evaluation:**  
  - Measures RMSE and provides visualizations comparing actual vs predicted values.

- Combines the feature extraction of encoder with a feed forward MLP.


---

## How to Run

### LSTM Approach

- **Training:**
  ```bash
  python lstm_approach.py --train
  ```
- **Prediction:**
  ```bash
  python lstm_approach.py --predict
  ```

### ARIMA Approach

- arima_approach.ipynb and run the cells sequentially.

### Hybrid Approach

- **Training:**
  ```bash
  python hybrid_approach.py --train
  ```
- **Prediction:**
  ```bash
  python hybrid_approach.py --predict
  ```

---

## Dependencies

The project uses the following Python libraries:

- NumPy
- Pandas
- Scikit-learn
- TensorFlow
- Matplotlib
- Statsmodels
- Plotly
- PMDARIMA
