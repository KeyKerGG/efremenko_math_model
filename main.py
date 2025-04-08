import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

seed = int(input("Введите зерно для генерации случайных точек:"))
np.random.seed(seed)
num_cities = 10
cities = np.random.rand(num_cities, 2) * 100

distances = cdist(cities, cities, metric='euclidean')

def plot_route(cities, route):
    route_cities = np.concatenate([route, route[:1]])
    plt.figure(figsize=(10, 6))
    plt.plot(cities[route_cities, 0], cities[route_cities, 1], 'o-', markersize=8, label="Маршрут")
    plt.scatter(cities[:, 0], cities[:, 1], c='red', marker='x', label='Города')
    for i, (x, y) in enumerate(cities):
        plt.text(x + 1, y + 1, f'{i}', fontsize=12, color='black')
    plt.title('Маршрут коммивояжёра')
    plt.legend()
    plt.grid(True)
    plt.show()

from scipy.optimize import linear_sum_assignment

row_ind, col_ind = linear_sum_assignment(distances)

plot_route(cities, row_ind)


import time


# Функция для вычисления длины маршрута
def calculate_route_length(route, dist_matrix):
    return sum(dist_matrix[route[i], route[i + 1]] for i in range(len(route) - 1)) + dist_matrix[route[-1], route[0]]


# Варьируем количество городов и измеряем время
city_counts = range(5, 21)
route_lengths = []
execution_times = []

for num_cities in city_counts:
    cities = np.random.rand(num_cities, 2) * 100
    distances = cdist(cities, cities, metric='euclidean')

    start_time = time.time()
    row_ind, col_ind = linear_sum_assignment(distances)
    execution_times.append(time.time() - start_time)

    route_length = calculate_route_length(row_ind, distances)
    route_lengths.append(route_length)

# Строим график
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(city_counts, route_lengths, marker='o')
plt.title("Зависимость длины маршрута от числа городов")
plt.xlabel("Число городов")
plt.ylabel("Длина маршрута")

plt.subplot(1, 2, 2)
plt.plot(city_counts, execution_times, marker='o', color='red')
plt.title("Зависимость времени вычисления от числа городов")
plt.xlabel("Число городов")
plt.ylabel("Время (сек)")

plt.tight_layout()
plt.show()
