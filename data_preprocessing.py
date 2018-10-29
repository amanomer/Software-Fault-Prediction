#load file
import xlrd
import imblearn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def displaystats( nsheet ):
    rows = nsheet.nrows
    cols = nsheet.ncols
    posi = 0
    neg = 0
    for x in range(1,rows):
        if nsheet.cell(x,cols-1).value == "true":
            posi = posi + 1
        else:
            neg = neg + 1
    total = posi + neg
    pcent = (posi / total) * 100
    ncent = (neg / total) * 100
    print ("Ture: " + str(posi) + " = "+ "{0:.2f}".format(pcent) +"%\n")
    print ("False: " + str(neg) + " = "+ "{0:.2f}".format(ncent) +"%\n")

loc = ("D:/CM1/")
rdfile = ("final2db.xls")
wb = xlrd.open_workbook( loc + rdfile )
sheet = wb.sheet_by_index(0)

#displaystats(sheet)

xls_file = pd.ExcelFile( loc + rdfile )
df = xls_file.parse( "CM" )
print(df["defects"].value_counts())
#print(df.iloc[0])


labels = df.columns[:]
X = df[labels]
y = df['defects']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


#print(df['defects'])
#sns.set_style("whitegrid");
#Code for histogram
'''sns.FacetGrid(df, hue="defects", size=4) \
    .map(sns.distplot, "lOCode") \
    .add_legend();
plt.show();'''
#if df["defects"] == "true":
    #print (df["defects"])
'''CM_true = df.loc[df["defects"] == "true"];
CM_false = df.loc[df["defects"] == "false"];

plt.plot(CM_true["loc"], np.zeros_like(CM_true['loc']), 'o')

plt.show()'''


'''from imblearn.under_sampling import TomekLinks

labels = df.columns[:]
X = df[labels]
y = df['defects']
tl = TomekLinks(return_indices=True, ratio='majority')
X_tl, y_tl, id_tl = tl.fit_sample(X, y)

#print('Removed indexes:', id_tl)
print('After Tomek links under-sampling\n')
print(y_tl)
plot_2d_space(X_tl, y_tl, 'Tomek links under-sampling')'''

