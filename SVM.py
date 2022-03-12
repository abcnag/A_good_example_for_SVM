import pandas as  pd
import numpy as np
import matplotlib.pyplot as plt
###
df = pd.read_csv("C:/Users/M/Desktop/abbas/learning me/MakTab Khoone/Data Sets/heart.csv")
data = df[['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh',
            'exng', 'oldpeak', 'slp', 'caa', 'thall']]
###
from sklearn.preprocessing import StandardScaler
sclaer = StandardScaler().fit(data)
data = sclaer.transform(data)
###
from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(data,df['output'])
###
# loss func
def loss(y_pre,y_test):
    correct_pre  = 0
    for i in range(len(y_pre)):
        if y_pre[i] == y_test[i]:
            correct_pre += 1
    print('model Accuracy ===>' , 100*correct_pre/len(y_pre) , "%")
###
# SVM
from sklearn.svm import SVC
svc = SVC(C=1 , kernel='rbf' , gamma=0.01)
'''
C: Regularization parameter
kernel : Specifies the kernel type to be used in the algorithm
1^^^ linear : The most basic way to use a SVC is with a linear kernel. it is not very good
2^^^ sigmoid : 1/(1+e^x) --> 0,1
3^^^ poly : in the polynomial kernel, we simply calculate the dot product by increasing the power of the kernel
4^^^ rbf : the squared Euclidean distance between two data points 

gamma : the decision region (The higher it is, the smaller the radius of each point)
C : The higher it is, the higher the penalty for each error
'''
svc.fit(X_train,Y_train)

# find loss
Y_pre5 = svc.predict(X_test)
loss(Y_pre5,Y_test.values)