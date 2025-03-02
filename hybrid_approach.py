import numpy as np
import pandas as pd
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os 
import argparse

def prepare_data():
    
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
    train, valid, Y_train, Y_valid = train_test_split(data_series, labels.values, test_size=0.1, random_state=0, shuffle=False)
    print("Train set", train.shape)
    print("Validation set", valid.shape)
    train.head()
    X_train = train.values.reshape((train.shape[0], train.shape[1], 1))
    X_valid = valid.values.reshape((valid.shape[0], valid.shape[1], 1))

    return X_train, X_valid, Y_train, Y_valid

def build_encoder_decoder(timesteps, n_features=1):
    model = models.Sequential([
        layers.LSTM(64, return_sequences=True, input_shape=(timesteps, n_features)),
        layers.LSTM(32, return_sequences=True),
        layers.LSTM(16),
        layers.RepeatVector(timesteps),
        layers.LSTM(32, return_sequences=True),
        layers.TimeDistributed(layers.Dense(1))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# def visualization(mlp_history):
#     fig, ax = plt.subplots(figsize=(22, 7))

#     ax.plot(mlp_history.history['loss'], label='Train loss')
#     ax.plot(mlp_history.history['val_loss'], label='Validation loss')
#     ax.legend(loc='best')
#     ax.set_title('MLP with LSTM encoder')
#     ax.set_xlabel('Epochs')
#     ax.set_ylabel('MSE')

#     plt.show()

def visualization(mlp_history, save_path='mlp_history.png'):
    fig, ax = plt.subplots(figsize=(22, 7))
    
    ax.plot(mlp_history.history['loss'], label='Train loss')
    ax.plot(mlp_history.history['val_loss'], label='Validation loss')
    ax.legend(loc='best')
    ax.set_title('Regular LSTM')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('MSE')
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def build_mlp(input_dim):
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_dim=input_dim),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(X_train, y_train, X_val, y_val, epochs=20, batch_size=128):
    # encoder-decoder
    encoder_decoder = build_encoder_decoder(X_train.shape[1])
    encoder_decoder.fit(X_train, X_train,
                       epochs=epochs,
                       batch_size=batch_size,
                       validation_data=(X_val, X_val))
    
    # encoder model
    encoder = models.Model(encoder_decoder.inputs,
                          encoder_decoder.layers[2].output)
    
    # encoded features
    train_encoded = encoder.predict(X_train)
    val_encoded = encoder.predict(X_val)
    
    # hybrid features
    X_train_hybrid = np.concatenate([
        train_encoded, 
        X_train[:, -1, :]  # Last months' sales
    ], axis=1)
    
    X_val_hybrid = np.concatenate([
        val_encoded,
        X_val[:, -1, :]
    ], axis=1)
    
    # train MLP
    mlp = build_mlp(X_train_hybrid.shape[1])
    history = mlp.fit(X_train_hybrid, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_val_hybrid, y_val))
    
    return encoder, mlp, history

def save_models(encoder, mlp, encoder_path='encoder_model.keras', mlp_path='mlp_model.keras'):
    encoder.save(encoder_path)
    mlp.save(mlp_path)
    print(f"Models saved to {encoder_path} and {mlp_path}")

def load_models(encoder_path='encoder_model.keras', mlp_path='mlp_model.keras'):
    encoder = load_model(encoder_path)
    mlp = load_model(mlp_path)
    print(f"Models loaded from {encoder_path} and {mlp_path}")
    return encoder, mlp

def predict(encoder, mlp, X_data):
    # encoded features
    encoded = encoder.predict(X_data)
    
    # hybrid features
    hybrid_features = np.concatenate([
        encoded,
        X_data[:, -1, :]  # Last months' sales
    ], axis=1)
    
    return mlp.predict(hybrid_features).flatten()

def visualize_results(actual, predicted, save_path="predictions.png"):
    plt.figure(figsize=(15, 6))
    plt.plot(actual, label='Actual', marker='o')
    plt.plot(predicted, label='Predicted', marker='x', linestyle='--')
    plt.title('Validation Results')
    plt.xlabel('Time Step')
    plt.ylabel('Item Count')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Sales Prediction Model')
    parser.add_argument('--train', action='store_true', help='Train new models')
    parser.add_argument('--predict', action='store_true', help='Run predictions')
    args = parser.parse_args()

    X_train, X_valid, Y_train, Y_valid = prepare_data()

    if args.train:
        encoder, mlp, history = train_model(X_train, Y_train, X_valid, Y_valid)
        save_models(encoder, mlp)
        visualization(history)

    if args.predict:
        encoder, mlp = load_models()
        val_pred = predict(encoder, mlp, X_valid)
        val_rmse = np.sqrt(mean_squared_error(Y_valid, val_pred))
        print(f'Validation RMSE: {val_rmse:.4f}')
        visualize_results(Y_valid, val_pred)

if __name__ == "__main__":
    main()