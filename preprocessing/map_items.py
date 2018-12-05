import csv

filename = "../data/ratings.csv"

# maps itemId from real id to id 0-6000
itemId_map = {}

unique_itemIds = set()
unique_userIds = set()

# userId, itemId, rating, <other ignored columns>
rows = []
with open(filename) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    row_num = 0
    for row in readCSV:
        if row_num > 0:
            userId = int(row[0])
            itemId = int(row[1])
            rating = float(row[2])

            rows.append([userId, itemId, rating])

            unique_itemIds.add(itemId)
            unique_userIds.add(userId)
        row_num += 1

# For each unique item id, map to next available id starting from 1
itemIds = sorted(list(unique_itemIds))
next_avail_id = 1
for id in  itemIds:
    itemId_map[id] = next_avail_id
    next_avail_id += 1

# Update itemId in each row to be between 0-9000
for row in rows:
    row[1] = itemId_map[row[1]]

filename = '../data/ratings_mapped.csv'
with open(filename, "w", newline='') as file:
    file.write("userId,itemId,rating\n")
    for row in rows:
        row = [str(i) for i in row]
        line = ",".join(row)
        file.write(line)
        file.write('\n')

