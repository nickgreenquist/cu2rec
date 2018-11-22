import numpy as np
import csv

filename = "../data/test/test_ratings3"
# Load in sparse ratings as dense matrix (hardcode dimensions for now)
rows = 8
cols = 5
n_factors = 2
ratings = np.zeros(shape=(rows, cols), dtype=np.float)
with open("{}.csv".format(filename)) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    row_num = 0
    for row in readCSV:
        if row_num > 0:
            u = int(row[0])
            i = int(row[1])
            rating = float(row[2])
            ratings[u - 1, i - 1] = rating
        row_num += 1

# Load in all components
P = np.loadtxt("{}_f{}_{}.csv".format(filename, n_factors, "p"), delimiter=',')
Q = np.loadtxt("{}_f{}_{}.csv".format(filename, n_factors, "q"), delimiter=',')
user_bias = np.loadtxt("{}_f{}_{}.csv".format(filename, n_factors, "user_bias"), delimiter=',')
item_bias = np.loadtxt("{}_f{}_{}.csv".format(filename, n_factors, "item_bias"), delimiter=',')
global_bias = np.loadtxt("{}_f{}_{}.csv".format(filename, n_factors, "global_bias"), delimiter=',').item()


# Calculate all missing ratings and output
print("Orig Ratings:")
print(ratings)
print()
for u in range(rows):
    for i in range(cols):
        pred = global_bias + user_bias[u] + item_bias[i] + np.dot(P[u], Q[i])
        ratings[u][i] = pred

np.set_printoptions(precision=2)
print("Predicted Ratings:")
print(ratings)
