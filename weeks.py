from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pandas_datareader.data import DataReader
import matplotlib.pyplot as plt

# ==============================================================================================
# beg: config
# ==============================================================================================
STOCK_NAME = 'EFX'#'KIRK'
YEARS = 20
END_DATE = datetime.now()
BEG_DATE = datetime(END_DATE.year - YEARS, END_DATE.month, END_DATE.day)

WIN_SIZE = 64
TRAIN_TEST_RAT = 0.7

EPOCHS = 5
BATCH_SIZE = 512#1
# ==============================================================================================
# end: config
# ==============================================================================================


def get_stock(stock, beg, end, write=False, load=False):
    """ return df from yfinance """
    if load ==True: return pd.read_csv(f'./{stock}.csv')

    ret_df = DataReader(stock, 'yahoo', beg, end)
    if write: ret_df.to_csv(f'./{stock}.csv')
    return ret_df 

data = get_stock(
    stock=STOCK_NAME, 
    beg=BEG_DATE, 
    end=END_DATE,
    load=False, 
    write=False)

close_data = data.filter(['Close'])


PREV_DATA_SERIES_LEN = 64
PREV_DATA_IDX = -PREV_DATA_SERIES_LEN

prev_data = close_data[PREV_DATA_IDX:].values.flatten()


from realtime import pred_next
from keras.models import load_model
import joblib

model = load_model(f'./{STOCK_NAME}.h5')
scaler = joblib.load(f'./{STOCK_NAME}_scaler.bin')

next_preds = pred_next(
    time_steps=7,
    model=model,
    start_series=prev_data.reshape(1, 64, 1),
    series_len=64
)
next_preds = scaler.inverse_transform(
    np.array(next_preds)
        .reshape(-1, 1))\
        .flatten()

next_days = pd.date_range(
    datetime.now()+timedelta(1), 
    datetime.now()+timedelta(7), 
    freq='D')

pred_data = pd.DataFrame({
    'date': next_days, 
    'PredictedClose': next_preds
    }).set_index('date')


# Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(close_data['Close'][PREV_DATA_IDX:])
plt.plot(pred_data['PredictedClose'])
#plt.plot(valid[['Close', 'Predictions']])
#plt.plot(hdout[['Close', 'Predictions']])
#plt.legend(['Train', 'Val', 'Val Preds', 'Holdout', 'Holdout Preds'], loc='lower right')
plt.show()