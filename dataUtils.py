import pandas as pd
import numpy as np

def getAttijariDataset() :
    dfs = pd.read_html('data.aspx', header=0)
    data = dfs[0]
    data.columns = ['date', 'close_price', 'adj_price', 'evol', 'quantity', 'volume']
    data = data.drop(['adj_price', 'evol', 'quantity', 'volume'], 1)
    data['close_price'] = data['close_price'] / 100
    data.index = pd.to_datetime(data['date'])
    data = data.drop(['date'],1)
    print(data.info())
    return data


def next_batch(training_data, batch_size, steps):
    # Grab a random starting point for each batch
    rand_start = np.random.randint(0, len(training_data) - steps)

    # Create Y data for time series in the batches
    y_batch = np.array(training_data[rand_start:rand_start + steps + 1]).reshape(1, steps + 1)

    return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1)