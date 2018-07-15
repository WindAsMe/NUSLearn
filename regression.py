import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


if __name__ == '__main__':
    boston = load_boston()
    # print boston.DESCR
    bos = pd.DataFrame(boston.data)
    bos.columns = boston.feature_names
    X = bos['LSTAT'].values.reshape(-1, 1)
    Y = bos['AGE'].values.reshape(-1, 1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
    # print X.shape
    lm = LinearRegression()
    lm.fit(X, Y)
    # print(lm.predict(23))
    Y_pred = lm.predict(X_test)
    # print "MAE: {} ".format(metrics.mean_absolute_error(Y_test, Y_pred))
    newX = np.column_stack((bos['CRIM'], bos['LSTAT']))
    newX_train, newX_test, Y_train, Y_test = train_test_split(newX, Y, test_size=0.33)
    lm.fit(newX_train, Y_train)
    Y_pred = lm.predict(newX_test)
    # print 'MAE: {}'.format(metrics.mean_absolute_error(Y_test, Y_pred))
    mse = metrics.mean_squared_error(Y_test, Y_pred)
    # print 'MSE: {}'.format(mse)
    # print 'RMSE: {}'.format(np.sqrt(mse))
