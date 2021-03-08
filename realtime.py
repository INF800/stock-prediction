import numpy as np

def pred_next(time_steps: int, model, start_series, series_len=64):

    accumulator = []

    # start_series::(1, series_len, 1)
    cur_out = model.predict(start_series).flatten()[0]
    accumulator.append(cur_out)
    
    i = 0
    while i < time_steps-1:

        if i==0:
            next_series = np.append(start_series.flatten()[1:], cur_out)        
        else:
            next_series = np.append(next_series.flatten()[1:], cur_out)

        cur_out = model.predict(next_series.reshape(1, series_len, 1)).flatten()[0]
        accumulator.append(cur_out)

        i += 1

    return accumulator


if __name__ == '__main__':


    from keras.models import Sequential
    from keras.layers import Dense, LSTM
    from keras.models import load_model

    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape= (64, 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model = load_model('temp.h5')

    # # prev 64 days weeks
    # start_series = np.zeros((1, 64, 1))
    # out = model.predict(start_series).flatten()[0]
    # 
    # next_series = np.append(start_series.flatten()[1:], out)
    # out = model.predict(next_series.reshape(1, 64, 1)).flatten()[0]

    prediction = pred_next(
        time_steps=7,
        model=model,
        start_series=np.zeros((1, 64, 1)),
        series_len=64
    )

    # print(prediction)
    # print(len(prediction))
    # [0.028803729, 0.029868305, 0.031743404, 0.034207754, 0.03708552, 0.04024382, 0.043586105]
    # 7