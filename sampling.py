import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def plot_2d_space(X, y, label='Classes'):   
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()

def show_distribution( y ):
    f=0
    t=0
    
    for x in y:
        #print(type(x))
        if x:
            t = t+1
        else: 
            f = f+1
            
    print('False = ' + str(f))
    print('True = ' + str(t))

loc = ("D:/CM1/")
rdfile = ("final2db.xls")
xls_file = pd.ExcelFile( loc + rdfile )
df = xls_file.parse( "CM" )
print(df["defects"].value_counts())

'''from imblearn.under_sampling import TomekLinks

labels = df.columns[:]
X = df[labels]
y = df['defects']
tl = TomekLinks(return_indices=True, ratio='majority')
X_tl, y_tl, id_tl = tl.fit_sample(X, y)

print('\nAfter Tomek links under-sampling\n')
#print(y_tl)
show_distribution(y_tl)'''
#plot_2d_space(X_tl, y_tl, 'Tomek links under-sampling')

'''from imblearn.under_sampling import ClusterCentroids

cc = ClusterCentroids(ratio={0: 10})
X_cc, y_cc = cc.fit_sample(X, y)
print('\nAfter Cluster Centroids under-sampling\n')
show_distribution(y_cc)'''
#plot_2d_space(X_cc, y_cc, 'Cluster Centroids under-sampling')

from imblearn.over_sampling import SMOTE

smote = SMOTE(ratio='minority')
X_sm, y_sm = smote.fit_sample(X, y)
print('\nAfter SMOTE over-sampling\n')
show_distribution(y_sm)

from imblearn.combine import SMOTETomek

smt = SMOTETomek(ratio='auto')
X_smt, y_smt = smt.fit_sample(X, y)
print('\nAfter SMOTE over-sampling followed by Tomek link under-sampling\n')
show_distribution(y_smt)

X_train, X_test, y_train, y_test = train_test_split(X_smt, y_smt, test_size=0.2, random_state=1)

#model = XGBClassifier()
nn = 4
model = KNeighborsClassifier(n_neighbors=nn)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
#show_distribution(y_train)
#show_distribution(y_test)
#show_distribution(y_pred)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))