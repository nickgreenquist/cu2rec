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

# Load in all pre_trained components
Q = np.loadtxt("{}_f{}_{}.csv".format(filename, n_factors, "q"), delimiter=',')
item_bias = np.loadtxt("{}_f{}_{}.csv".format(filename, n_factors, "item_bias"), delimiter=',')
global_bias = np.loadtxt("{}_f{}_{}.csv".format(filename, n_factors, "global_bias"), delimiter=',').item()

# create new user
new_user = np.array([1.0,1.0,0.0,5.0,0.0])

# partial fit a new P and user_bias using trained Q and item_bias
learning_rate = 0.1
user_bias_reg = 0.1
P_reg = 0.1

# 1. get the user_bias for this user
new_user_bias = np.mean(new_user) - global_bias

# 2. set up new random P
mu, sigma = 0, 0.1
new_user_P = s = np.random.normal(mu, sigma, n_factors)

# 3. computer small number of iterations of SGD
for iteration in range(5):
    # 3.1 calculate loss
    errors = np.zeros(shape=(cols), dtype=np.float)
    for i in range(cols):
        rating = new_user[i]
        if rating != 0.0:
            pred = global_bias + new_user_bias + item_bias[i] + np.dot(new_user_P, Q[i])
            errors[i] = rating - pred

    # 3.2 calculate total loss and output
    total_loss = 0.0
    for j in range(cols):
        total_loss += pow(errors[j], 2)
    print("Loss at Iteration {}: {}".format(iteration, total_loss))

    # 3.3 run single SGD iteration
    new_user_bias_target = new_user_bias
    new_user_P_target = np.copy(new_user_P)
    for i in range(cols):
        for f in range(n_factors):
            ub_update = learning_rate * (errors[i] - user_bias_reg * new_user_bias)
            new_user_bias_target += ub_update

            p_update = learning_rate * (errors[i] * Q[i][f] - P_reg * new_user_P[f])
            new_user_P_target[f] += p_update
    
    # 3.4 copy updated components back to original
    new_user_P = np.copy(new_user_P_target)
    new_user_bias = new_user_bias_target

# 4. Use new components to get recs
print("\nOriginal Ratings: ")
print(new_user)
print()

for i in range(cols):
    pred = global_bias + new_user_bias + item_bias[i] + np.dot(new_user_P, Q[i])
    new_user[i] = pred

print("Predicted Ratings:")
print(new_user)