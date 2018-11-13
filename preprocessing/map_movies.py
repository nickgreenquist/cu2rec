import csv

filename = "../data/ratings.csv"

# maps movieId from real id to id 0-6000
movieId_map = {}

unique_movieIds = set()
unique_userIds = set()

# userId,movieId,rating,timestamp
rows = []
with open(filename) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    row_num = 0
    for row in readCSV:
        if row_num > 0:
            userId = int(row[0])
            movieId = int(row[1])
            rating = float(row[2])
            timestamp = int(row[3])

            rows.append([userId, movieId, rating, timestamp])

            unique_movieIds.add(movieId)
            unique_userIds.add(userId)
        row_num += 1

# For each unique movie id, map to next available id starting from 1
movieIds = sorted(list(unique_movieIds))
next_avail_id = 1
for id in  movieIds:
    movieId_map[id] = next_avail_id
    next_avail_id += 1

# Update movieId in each row to be between 0-9000
for row in rows:
    row[1] = movieId_map[row[1]]

filename = '../data/ratings_mapped.csv'
with open(filename, "w", newline='') as file:
    file.write("userId,movieId,rating,timestamp\n")
    for row in rows:
        row = [str(i) for i in row]
        line = ",".join(row)
        file.write(line)
        file.write('\n')

