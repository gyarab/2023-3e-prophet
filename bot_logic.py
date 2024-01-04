import ccxt

#pytorch imports for neural network
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


exchange = ccxt.coinex({
        'apiKey': '01566309D7D54F6B83CD7BD57090B485',
        'secret': 'FF1A020B10EC704972C034475F2BBA140F814F87B401F37B',
    })
def neco_cinskyho():
    # load the dataset, split into input (X) and output (y) variables
    dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',') #loads as numpy array
    X = dataset[:,0:8] #in all rows columns 0-7
    y = dataset[:,8] #only last value (8th column) - will be 0 or 1

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)



    #Define Model
    model = nn.Sequential(
        nn.Linear(8, 12),
        nn.ReLU(),
        nn.Linear(12, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.Sigmoid())
    print(model)


    # Prepare for Training
    loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.002)


    # Training a Model
    n_epochs = 100 # Epoch: Passes the entire training dataset to the model once, number
    batch_size = 10 # Batch: One or more samples passed to the model, from which the gradient descent algorithm will be executed for one iteration

    for epoch in range(n_epochs):
        for i in range(0, len(X), batch_size):
            Xbatch = X[i:i+batch_size]
            y_pred = model(Xbatch)
            ybatch = y[i:i+batch_size]
            loss = loss_fn(y_pred, ybatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Finished epoch {epoch}, latest loss {loss}')

    # compute accuracy (no_grad is optional)
    with torch.no_grad():
        y_pred = model(X)

    accuracy = (y_pred.round() == y).float().mean()
    print(f"Accuracy {accuracy}")    


    # make probability predictions with the model
    predictions = model(X)
    # round predictions
    rounded = predictions.round()

    # make class predictions with the model
    predictions = (model(X) > 0.5).int()
    for i in range(5):
        print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))
def get_btc_price():
    
    # Fetch ticker information for BTC/USDT (you can change the symbol based on your needs)
    ticker = exchange.fetch_ticker('BTC/USDT')
    # Extract the last price from the ticker data
    last_price = ticker['last']
    
    return last_price
#returns account balance as dictoniary
def get_balance():
    
    # Fetch your account balance
    balance = exchange.fetch_balance()

    return balance

def get_markets():
    markets = exchange.load_markets()

    return list(markets.keys())

if __name__ == '__main__':
    # print(get_btc_price())
    print("CoinEx Account Balance:")
    print(get_balance())
    #print(get_btc_price())

