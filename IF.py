import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

cities = pd.read_csv("cities_2_decimal.csv")
data_cities = cities.values

names = data_cities[:,0]
X = data_cities[:,1]
Y = data_cities[:,2]

""" elimino la columna de nombres de ciudades """
data_cities_without_name = np.delete(data_cities, 0, axis=1)
data_cities_dataframe = pd.DataFrame(data_cities_without_name)
print(data_cities_dataframe)

""" isolation forest """
clf = IsolationForest(n_estimators=100, max_samples=100, contamination=.1) 
output = clf.fit(data_cities_without_name)
if_scores = clf.decision_function(data_cities_without_name)
if_anomalies=clf.predict(data_cities_without_name)
if_anomalies=pd.Series(if_anomalies).replace([-1,1],[1,0])
print(type(if_anomalies))
if_anomalies=data_cities_dataframe[if_anomalies==1]
print(if_anomalies)


plt.scatter(X,Y,c='white',s=20,edgecolor='k')
plt.scatter(if_anomalies.iloc[:,0],if_anomalies.iloc[:,1],c='red')
plt.xlabel('Income')
plt.ylabel('Spend_Score')
plt.title('Isolation Forests - Anomalies')
plt.show()

print(if_anomalies)
plt.figure(figsize=(12,8))
plt.hist(if_scores)
plt.title("IF")
plt.show()