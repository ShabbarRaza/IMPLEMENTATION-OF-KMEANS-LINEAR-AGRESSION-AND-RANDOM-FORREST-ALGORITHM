# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wiYheCZXg86BlV3wDvccXGTNDrSCG-Po
"""

import itertools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn import datasets

from sklearn.linear_model import LogisticRegression
from sklearn import datasets, linear_model
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score, train_test_split

from mlxtend.plotting import plot_learning_curves
from mlxtend.plotting import plot_decision_regions
df = pd.read_csv(r'C:\Users\SHABBAR RAZA\Desktop\lenses.csv' )
df.head()

df = pd.read_csv(r'C:\Users\SHABBAR RAZA\Desktop\lenses.csv' )
df.head()
df.columns  = ['hard contact lense', 'soft contact lense', 'no need', 'age', 'spectacle prespriction', 'astigmatic', 'fear production rate']

one_hot_data = pd.get_dummies(df['hard contact lense', 'soft contact lense', 'no need', 'age', 'spectacle prespriction', 'astigmatic', 'fear production rate']
 )


X, y = df.data[:, 1:3], df.target

clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = regr = linear_model.LinearRegression()
clf4 = tree.DecisionTreeClassifier()
lr = LogisticRegression()
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3, clf4], 
                          meta_classifier=lr)

label = ['KNN', 'Random Forest', 'Linear Regression', 'Desicion Tree' 'Stacking Classifier']
clf_list = [clf1, clf2, clf3, clf4, sclf]
    
fig = plt.figure(figsize=(10,8))
gs = gridspec.GridSpec(2, 2)
grid = itertools.product([0,1],repeat=2)

clf_cv_mean = []
clf_cv_std = []
for clf, label, grd in zip(clf_list, label, grid):
        
    scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
    print "Accuracy: %.2f (+/- %.2f) [%s]" %(scores.mean(), scores.std(), label)
    clf_cv_mean.append(scores.mean())
    clf_cv_std.append(scores.std())
        
    clf.fit(X, y)
    ax = plt.subplot(gs[grd[0], grd[1]])
    fig = plot_decision_regions(X=X, y=y, clf=clf)
    plt.title(label)