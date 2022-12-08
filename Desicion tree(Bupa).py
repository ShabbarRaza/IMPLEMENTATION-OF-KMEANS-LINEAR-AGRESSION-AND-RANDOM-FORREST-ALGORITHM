{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.tree import DecisionTreeClassifier \n",
    "from sklearn.model_selection import train_test_split\n",
    "import pydotplus\n",
    "from IPython.display import image\n",
    "\n",
    "df = pd.read_csv(r'C:\\Users\\SHABBAR RAZA\\Desktop\\bupa.csv' )\n",
    "df.head()\n",
    "df.columns  = ['mcv', 'alkphos', 'sgpt', 'sgot', 'gammagt', 'drinkno#', 'selector']\n",
    "\n",
    "one_hot_data = pd.get_dummies(df['mcv', 'alkphos', 'sgpt', 'sgot', 'gammagt', 'drinkno#', 'selector'])\n",
    " \n",
    "clf = tree.DecisionTreeClassifier()\n",
    "clf_train = clf.fit(one_hot_data, df['mcv', 'alkphos'])\n",
    "\n",
    "\n",
    "clf = tree.DecisionTreeClassifier()\n",
    "clf_train = clf.fit(one_hot_data, df['mcv', 'alkphos'])\n",
    "\n",
    "\n",
    "print(tree.export_graphviz(clf_train, None))\n",
    "\n",
    "\n",
    "dot_data = tree.export_graphviz(clf_train, out_file=None, feature_names=list(one_hot_data.columns.values), \n",
    "                                class_names=['selector'], rounded=True, filled=True) \n",
    "\n",
    "graph = pydotplus.graph_from_dot_data(dot_data)\n",
    "\n",
    "Image(graph.create_png())\n",
    "\n",
    "\n",
    "clf = tree.DecisionTreeClassifier        \n",
    "clf_train = clf.fit(one_hot_data,(df))\n",
    "prediction = clf_train.predict(0,0,1,1,0,1,1,0,1,0,0,1,1,0,1,1,0,1)    \n",
    "print(prediction) \n",
    "['high']\n",
    "\n",
    "prediction = clf_train.predict(0,0,0,0,1,1,0,1,0,0,1,1,0,1,0,0,0)    \n",
    "print(prediction) \n",
    "['low']"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
