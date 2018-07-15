import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn
from sklearn.datasets import load_boston

if __name__ == '__main__':
    boston=load_boston()
    # print boston.DESCR
    bos = pd.DataFrame(boston.data)
    bos.columns = boston.feature_names
    #print(bos['CRIM'].corr(bos['LSTAT']))
    # print(bos.head())
    # bos['PRICE']=boston.target
    print(bos.head())