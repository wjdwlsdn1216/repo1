import pandas as pd
from sklearn import svm, metrics
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

my_input = np.arange(30).reshape(6,5)

my_df = pd.DataFrame(my_input)
my_data = my_df[[0,1,3]]
my_label = my_df[2]

clf = KNeighborsClassifier(n_neighbors=2)
clf.fit(my_data,my_label)
pre =clf.predict(my_data)

ac_score = metrics.accuracy_score(my_label, pre)
print("정답률 : ", ac_score)