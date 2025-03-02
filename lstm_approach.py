import numpy as np
import pandas as pd
import os
import argparse
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout
from sklearn.model_selection import train_test_split
import tensorflow as tf 
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def prepare_lstm_data():
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    test = pd.read_csv(os.path.abspath(os.path.join(dir_path,"future_sales"))+'/test.csv', dtype={'ID': 'int32', 'shop_id': 'int32', 
                                                  'item_id': 'int32'})
    item_categories = pd.read_csv(os.path.abspath(os.path.join(dir_path,"future_sales"))+'/item_categories.csv', 
                                dtype={'item_category_name': 'str', 'item_category_id': 'int32'})
    items = pd.read_csv(os.path.abspath(os.path.join(dir_path,"future_sales"))+'/items.csv', dtype={'item_name': 'str', 'item_id': 'int32', 
                                                    'item_category_id': 'int32'})
    shops = pd.read_csv(os.path.abspath(os.path.join(dir_path,"future_sales"))+'/shops.csv', dtype={'shop_name': 'str', 'shop_id': 'int32'})
    sales = pd.read_csv(os.path.abspath(os.path.join(dir_path,"future_sales"))+'/sales_train.csv', parse_dates=['date'], 
                        dtype={'date': 'str', 'date_block_num': 'int32', 'shop_id': 'int32', 
                        'item_id': 'int32', 'item_price': 'float32', 'item_cnt_day': 'int32'})

    train = sales.join(items, on='item_id', rsuffix='_').join(shops, on='shop_id', rsuffix='_').join(item_categories, on='item_category_id', rsuffix='_').drop(['item_id_', 'shop_id_', 'item_category_id_'], axis=1)

    print(f'Train rows: {train.shape[0]}')
    print(f'Train columns: {train.shape[1]}')
    train['date'] = pd.to_datetime(train['date'], format='%d.%m.%Y')

    print(f"Min date from train set: {train['date'].min().date()}")
    print(f"Max date from train set: {train['date'].max().date()}")

    test_shop_ids = test['shop_id'].unique()
    test_item_ids = test['item_id'].unique()
    train = train[train['shop_id'].isin(test_shop_ids)]
    train = train[train['item_id'].isin(test_item_ids)]

    train_monthly = train[['date', 'date_block_num', 'shop_id', 'item_id', 'item_cnt_day']]
    train_monthly = train_monthly.sort_values('date').groupby(['date_block_num', 'shop_id', 'item_id'], as_index=False)
    train_monthly = train_monthly.agg({'item_cnt_day':['sum']})
    train_monthly.columns = ['date_block_num', 'shop_id', 'item_id', 'item_cnt']
    train_monthly = train_monthly.query('item_cnt >= 0 and item_cnt <= 20')
    train_monthly['item_cnt_month'] = train_monthly.sort_values('date_block_num').groupby(['shop_id', 'item_id'])['item_cnt'].shift(-1)

    train_monthly['month'] = train_monthly['date_block_num'] % 12
    train_monthly['year'] = (train_monthly['date_block_num'] // 12) + 2013
    
    monthly_series = train_monthly.pivot_table(index=['shop_id', 'item_id'], columns='date_block_num',values='item_cnt', fill_value=0).reset_index()
    monthly_series.head()

    first_month = 20
    last_month = 33
    serie_size = 12
    data_series = []

    for index, row in monthly_series.iterrows():
        for month1 in range((last_month - (first_month + serie_size)) + 1):
            serie = [row['shop_id'], row['item_id']]
            for month2 in range(serie_size + 1):
                serie.append(row[month1 + first_month + month2])
            data_series.append(serie)

    columns = ['shop_id', 'item_id']
    [columns.append(i) for i in range(serie_size)]
    columns.append('label')

    data_series = pd.DataFrame(data_series, columns=columns)
    data_series.head()

    data_series = data_series.drop(['item_id', 'shop_id'], axis=1)

    labels = data_series['label']
    data_series.drop('label', axis=1, inplace=True)
    train, valid, Y_train, Y_valid = train_test_split(data_series, labels.values, test_size=0.10, random_state=0, shuffle=False)
    print("Train set", train.shape)
    print("Validation set", valid.shape)
    train.head()
    X_train = train.values.reshape((train.shape[0], train.shape[1], 1))
    X_valid = valid.values.reshape((valid.shape[0], valid.shape[1], 1))

    return X_train, X_valid, Y_train, Y_valid

def build_lstm_model(input_shape):
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True, input_shape=input_shape)),
        Dropout(0.4),
        Bidirectional(LSTM(64)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(loss='mse', optimizer='adam')
    return model

def train_model():
    X_train, X_valid, Y_train, Y_valid = prepare_lstm_data()
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_valid, Y_valid),
        epochs=30,
        batch_size=256,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)]
    )
    return model, history

# def visualization(lstm_history):
#     fig, ax = plt.subplots(figsize=(22, 7))

#     ax.plot(lstm_history.history['loss'], label='Train loss')
#     ax.plot(lstm_history.history['val_loss'], label='Validation loss')
#     ax.legend(loc='best')
#     ax.set_title('Regular LSTM')
#     ax.set_xlabel('Epochs')
#     ax.set_ylabel('MSE')

#     plt.show()

def visualization(lstm_history, save_path='lstm_history.png'):
    fig, ax = plt.subplots(figsize=(22, 7))
    
    ax.plot(lstm_history.history['loss'], label='Train loss')
    ax.plot(lstm_history.history['val_loss'], label='Validation loss')
    ax.legend(loc='best')
    ax.set_title('Regular LSTM')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('MSE')
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def visualize_results(actual, predicted, save_path="lstm_model.png"):
    
    fig, ax = plt.subplots(figsize=(22, 7))
    
    predicted = predicted.flatten() 
    ax.plot(actual, label='Actual', marker='o', linestyle='-', linewidth=2, markersize=8)
    ax.plot(predicted, label='Predicted', marker='x', linestyle='--', linewidth=2, markersize=8)
    
    ax.set_title('Actual vs Predicted Sales', fontsize=16)
    ax.set_xlabel('Time Step', fontsize=14)
    ax.set_ylabel('Item Count', fontsize=14)
    
    ax.legend(loc='best', fontsize=12)
    
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

def save_model(model, path=None):
    if path is None:
        path = os.path.join(os.getcwd(), 'lstm_model.keras')
    model.save(path)
    print(f"lstm model saved to {path}")

def load_trained_model(path):
    model = load_model(path)
    print(f"Model loaded from {path}")
    return model

def predict(lstm_model, X_train, X_valid, Y_train, Y_valid):
    lstm_train_pred = lstm_model.predict(X_train).flatten()
    lstm_val_pred = lstm_model.predict(X_valid).flatten()
    train_rmse = np.sqrt(mean_squared_error(Y_train, lstm_train_pred))
    val_rmse = np.sqrt(mean_squared_error(Y_valid, lstm_val_pred))
    print('Train RMSE:', train_rmse)
    print('Validation RMSE:', val_rmse)
    visualize_results(Y_valid, lstm_val_pred)

def main():
    parser = argparse.ArgumentParser(description='Sales Prediction with LSTM')
    parser.add_argument('--train', action='store_true', help='Train new LSTM model')
    parser.add_argument('--predict', action='store_true', help='Run predictions using trained model')
    args = parser.parse_args()

    X_train, X_valid, Y_train, Y_valid = prepare_lstm_data()
    lstm_model = None
    lstm_history = None

    if args.train:
        print("Training new LSTM model...")
        lstm_model, lstm_history = train_model()
        save_model(lstm_model, path='lstm_model.keras')
        
        if lstm_history:
            visualization(lstm_history, save_path='lstm_training_history.png')

    if args.predict:
        print("Making predictions...")
        if not lstm_model:
            lstm_model = load_trained_model('lstm_model.keras')

        lstm_train_pred = lstm_model.predict(X_train).flatten()
        lstm_val_pred = lstm_model.predict(X_valid).flatten()
        
        train_rmse = np.sqrt(mean_squared_error(Y_train, lstm_train_pred))
        val_rmse = np.sqrt(mean_squared_error(Y_valid, lstm_val_pred))
        print(f'Train RMSE: {train_rmse:.4f}')
        print(f'Validation RMSE: {val_rmse:.4f}')
        
        visualize_results(Y_valid, lstm_val_pred, save_path="lstm_predictions.png")

    if not args.train and not args.predict:
        print("Specify either --train to train model or --predict to predict")

if __name__ == "__main__":
    main()