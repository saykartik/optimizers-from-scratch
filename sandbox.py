from Models import LinearRegression, dummy_data_gen
from Optimizers import GradientDescent, Adam, SGDM, Adagrad, Adadelta, RMSProp
from collections import OrderedDict
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

synthetic_betas = [3, 1, 5, 9, 4, 7]


# X, y = dummy_data_gen(betas=synthetic_betas, noise_var=0.3, std_range=100)

def run_models(X, y, lrdict, batch_size=50, dont_run=[], epochs=500, epoch_loss=True):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    opts = OrderedDict({
        'GD': GradientDescent(lr=0.01),
        'SGD': GradientDescent(lr=0.01),
        'SGDM': SGDM(lr=0.01, gamma=0.9),
        'Adam': Adam(lr=0.01),
        'Adagrad': Adagrad(lr=0.01),
        'Adadelta': Adadelta(),
        'RMSProp': RMSProp(lr=0.01),
    })

    opts = {k: v for k, v in opts.items() if k not in dont_run}

    for opt_name, lr in lrdict.items():
        opts[opt_name].lr = lr

    res = pd.DataFrame()
    hist = pd.DataFrame()
    for opt_name, opt in opts.items():
        print("Running Optimizer: ", opt_name)

        if opt_name == 'GD':
            batch_size = None
        elif opt_name == 'SGD':
            batch_size = 1

        reg = LinearRegression(batch_size=batch_size, opt=opt, epochs=epochs)
        reg.fit(X_train, y_train, epoch_loss=epoch_loss)
        final_loss = reg.history['loss'][-1]
        final_betas = [i[0] for i in reg.betas]

        y_train_pred = reg.predict(X_train)
        try:
            train_r2 = round(r2_score(y_train, y_train_pred) * 100,2)
        except:
            train_r2 = None

        y_test_pred = reg.predict(X_test)

        try:
            test_r2 = round( r2_score(y_test, y_test_pred) * 100,2)
        except:
            test_r2 = None

        cols = ['opt', 'loss', 'train_r2', 'test_r2'] + ['c' + str(i + 1) for i in range(len(final_betas))]
        vals = [opt_name, final_loss, train_r2, test_r2] + final_betas
        metrics = OrderedDict(zip(cols, vals))
        res = res.append(metrics, ignore_index=True)

        hist = hist.append(pd.DataFrame({
            'epoch': np.arange(len(reg.history['loss'])),
            'loss': reg.history['loss'],
            'opt': [opt_name] * len(reg.history['loss'])
        }), ignore_index=True)

    return res, hist
