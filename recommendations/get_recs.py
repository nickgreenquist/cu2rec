import numpy as np
import csv

filename = "../data/test_ratings3"
# Load in sparse ratings as dense matrix (hardcode dimensions for now)
rows = 8
cols = 5
n_factors = 2
ratings = [[0.0]*cols for i in range(rows)]
with open("{}.csv".format(filename)) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    row_num = 0
    for row in readCSV:
        if row_num > 0:
            u = int(row[0])
            i = int(row[1])
            rating = float(row[2])
            ratings[u-1][i-1] = rating
        row_num += 1
ratings = np.array(ratings)

# Load in all components
P = []
with open("{}_{}_{}.csv".format(filename, n_factors, "p")) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        user_vec = [float(row[0]), float(row[1])]
        P.append(user_vec)

Q = []
with open("{}_{}_{}.csv".format(filename, n_factors, "q")) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        item_vec = [float(row[0]), float(row[1])]
        Q.append(user_vec)

user_bias = []
with open("{}_{}_{}.csv".format(filename, n_factors, "user_bias")) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        b = float(row[0])
        user_bias.append(b)

item_bias = []
with open("{}_{}_{}.csv".format(filename, n_factors, "item_bias")) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        b = float(row[0])
        item_bias.append(b)

global_bias = 0.0
with open("{}_{}_{}.csv".format(filename, n_factors, "global_bias")) as csvfile:
    b = csvfile.readline()
    global_bias = float(b)

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