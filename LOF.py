import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

cities = pd.read_csv("cities_2_decimal.csv")
data_cities = cities.values

names = data_cities[:,0]
X = data_cities[:,1]
Y = data_cities[:,2]

""" elimino la columna de nombres de ciudades """
data_cities_without_name = np.delete(data_cities, 0, axis=1)
print(data_cities_without_name)

plt.title("anomaly detection based on LOF")

""" crear puntos """
rng = np.random.RandomState(0)
colors = rng.rand(1019)
plt.scatter(X, Y, c=colors, s=3., label='datums')

""" nombres en los puntos """       
for i, label in enumerate(names):
    plt.annotate(label, (X[i], Y[i]), fontsize=4)

""" LOF """
clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
output = clf.fit_predict(data_cities_without_name)
print(clf.fit_predict(data_cities_without_name))
datums_scores = clf.negative_outlier_factor_
df = pd.DataFrame(datums_scores, columns = ['DS'])
print(df)
radius = (datums_scores.max() - datums_scores) / (datums_scores.max() - datums_scores.min())
plt.scatter(X, Y, s=1000 * radius, edgecolors='r',
            facecolors='none', label='Outlier scores')
plt.show()

""" plt.hist(datums_scores, bins = 10, color = "blue", rwidth=0.9)
plt.title("Histograma")
plt.xlabel("Números")
plt.ylabel("Frecuencia")
plt.show() """

