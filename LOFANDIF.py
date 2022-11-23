import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

""" obtener ciudades CSV """


def cities():
    cities = pd.read_csv("cities_2_decimal.csv")
    data_cities = cities.values
    return data_cities


""" elimino la columna de nombres de ciudades """


def coordinates_cities():
    data_cities_without_name = np.delete(cities(), 0, axis=1)
    return data_cities_without_name


""" obtener 1019 colores alazar """


def colors_points():
    rng = np.random.RandomState(0)
    colors = rng.rand(1019)
    return colors


def LOF():
    clf = LocalOutlierFactor(n_neighbors=5, contamination=0.1)
    output = clf.fit_predict(coordinates_cities())
    datums_scores = clf.negative_outlier_factor_
    anomalies = pd.DataFrame(output)[output == -1]
    cities_anomalies = pd.DataFrame(cities())[output == -1]
    return datums_scores, anomalies, cities_anomalies


def LOF_GRAPHIC(X, Y, colors, datums_scores,cities_anomalies):
    plt.title("LOF")
    plt.scatter(X, Y, c=colors, s=3.0, label="datums")
    radius = (datums_scores.max() - datums_scores) / (
        datums_scores.max() - datums_scores.min()
    )
    plt.scatter(
       cities_anomalies.iloc[:, 1], cities_anomalies.iloc[:, 2], c="red", edgecolor="k"
    )
    """ for i, label in enumerate(names):
        plt.annotate(label, (X[i], Y[i]), fontsize=4) """
    plt.show()


def LOF_HISTOGRAM(datums_scores):
    plt.hist(datums_scores, bins=10, color="blue", rwidth=0.9)
    plt.title("Histograma")
    plt.show()


def IF():
    clf = IsolationForest(n_estimators=10, max_samples=100, contamination=0.1)
    clf.fit(coordinates_cities())
    if_scores = clf.decision_function(coordinates_cities())
    if_anomalies = clf.predict(coordinates_cities())
    if_anomalies = pd.Series(if_anomalies).replace([-1, 1], [1, 0])
    data_cities_dataframe = pd.DataFrame(coordinates_cities())
    cities_anomalies = pd.DataFrame(cities())[if_anomalies == 1]
    if_anomalies = data_cities_dataframe[if_anomalies == 1]
    return if_scores, if_anomalies, cities_anomalies


def IF_GRAPHIC(X, Y, colors, if_anomalies):
    plt.title("IF")
    plt.scatter(X, Y, c=colors, s=20, edgecolor="k")
    plt.scatter(
        if_anomalies.iloc[:, 0], if_anomalies.iloc[:, 1], c="red", edgecolor="k"
    )
    plt.show()

def IF_HISTOGRAM(if_scores):
    plt.figure(figsize=(12, 8))
    plt.hist(if_scores)
    plt.title("IF")
    plt.show()


def DISPERSION_DIAGRAM(LOF, IF):
    plt.subplots()
    plt.scatter(LOF, IF, c="pink", edgecolor="k")
    plt.show()


names = cities()[:, 0]
X = cities()[:, 1]
Y = cities()[:, 2]

""" LOF """
datums_scores, anomalies, cities_anomalies = LOF()
""" LOF_GRAPHIC(X, Y, colors_points(), datums_scores,cities_anomalies)
LOF_HISTOGRAM(datums_scores)
print(cities_anomalies) """
""" print(cities_anomalies.to_string()) """


""" IF """
if_scores, if_anomalies, cities_anomalies = IF()
""" IF_GRAPHIC(X,Y,colors_points(),if_anomalies)
IF_HISTOGRAM(if_scores)
print(cities_anomalies) """
""" print(cities_anomalies.to_string()) """


""" DISPERSION DIAGRAM """
DISPERSION_DIAGRAM(datums_scores, if_scores)
