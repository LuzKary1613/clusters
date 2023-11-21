# Importación de bibliotecas:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Cargo de los datos:
data = pd.read_csv("C:/Users/Luz Karen/Desktop/clustering_data.csv")

# Visualización de datos:
plt.scatter(data['x'], data['y'])
plt.title('Distribución de Datos')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Inicialización y ajuste del modelo K-means:
kmeans = KMeans(n_clusters=3, init='random', random_state=42)
data['cluster'] = kmeans.fit_predict(data[['x', 'y']])

# Puntos de cada cluster
centroid_colors = ['blue', 'pink', 'purple']

# Visualización de resultados del clustering:
plt.scatter(data['x'], data['y'], c=data['cluster'], cmap='copper')
plt.title('Resultado del Clustering con K-means')
plt.xlabel('X')
plt.ylabel('Y')

# Visualización de centroides:
for i, color in zip(range(3), centroid_colors):
    cluster_center = kmeans.cluster_centers_[i]
    plt.scatter(cluster_center[0], cluster_center[1], s=200, c=color, label=f'Centroide {i+1}')

# Mostrar leyenda y gráfico final:
plt.legend()
plt.show()

# Imprimir los centroides:
print('Centroides:')
print(kmeans.cluster_centers_)
