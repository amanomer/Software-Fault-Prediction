import pandas as pd
import numpy as np
from scipy import stats

def outliers_z_score(ys):
    threshold = 3

    mean_y = np.mean(ys)
    stdev_y = np.std(ys)
    z_scores = [(y - mean_y) / stdev_y for y in ys]
    #return np.where(np.abs(z_scores) > threshold)
    return z_scores

#Loading Data
loc = ("D:/CM1/")
rdfile = ("final2db.xls")
xls_file = pd.ExcelFile( loc + rdfile )
df = xls_file.parse( "CM" )
print('Original Shape')
print(df.shape)

#Calculating Z-Score
df1 = df.values #dataframe to nparray
#z = np.abs(stats.zscore(df))
z = np.abs(outliers_z_score(df1))
#print(type(z))
df2 = df[(z < 3).all(axis=1)]
print('Outliers removed using z-score')
print(df2.shape)

q1 = df.quantile(0.25)
q3 = df.quantile(0.75)
IQR = q3 - q1
dff = df[~((df < (q1 - 1.5 * IQR)) | (df > (q3 + 1.5 * IQR))).any(axis=1)]
print('Outliers removed using IQR')
print(dff.shape)