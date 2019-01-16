"""
Decision tree for manufacturing data
Need to install graphviz to see decision-tree graph.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
import graphviz 

#file path
path_in = r'*******\bosch_small_data'

#Read by pandas
df_tr_cat = pd.read_csv(path_in + r'\train_cat.csv', low_memory = False)
df_tr_date = pd.read_csv(path_in + r'\train_date.csv', low_memory = False)
df_tr_num = pd.read_csv(path_in + r'\train_numeric.csv', low_memory = False)


"""
Test Decision Tree
"""

#Fill NA
df_tr_num_fna = df_tr_num.fillna(-10000)

#Target variable
df_tr_num_fna_tg = df_tr_num_fna.iloc[:,1:-1]

#Make Model
clf = tree.DecisionTreeClassifier(max_depth=4)
clf = clf.fit(df_tr_num_fna_tg, df_tr_num_fna["Response"])

#Make Graph
dot_data = tree.export_graphviz(clf, out_file=None, 
                      feature_names = df_tr_num_fna_tg.columns,
                      class_names = ["Res_0", "Res_1"],
                      filled=True, rounded=True,  
                      special_characters=True)  
graph = graphviz.Source(dot_data)  
graph



