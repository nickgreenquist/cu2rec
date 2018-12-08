"""Maps the Netflix ratings to a format readable by mf.cu.

Make sure you download the Netflix ratings from the link in the README.
"""

import csv


def map_rows(filename, user_mapping, item_mapping, add_missing=True):
    """Maps the rows of the file with user_mapping and item_mapping.

    If add_missing is True, it will automatically add the missing user and
    item ids to the mappings. They are mapped sequentially starting from 1.
    """
    # itemId, userId, rating, <other ignored columns>
    rows = []
    missing_users = 0
    missing_items = 0
    with open(filename) as csvfile:
        csv_rows = csv.reader(csvfile, delimiter=' ')
        for row in csv_rows:
            user_id = int(row[0])
            item_id = int(row[1])
            rating = float(row[3])  # There are two spaces before the ratings

            if user_id not in user_mapping:
                if add_missing:
                    user_mapping[user_id] = len(user_mapping) + 1
                else:
                    missing_users += 1
                    continue
            user_id = user_mapping[user_id]

            if item_id not in item_mapping:
                if add_missing:
                    item_mapping[item_id] = len(item_mapping) + 1
                else:
                    missing_items += 1
                    continue
            item_id = item_mapping[item_id]

            rows.append([user_id, item_id, rating])
    if missing_users > 0:
        print("Skipped %d rows because of missing users" % missing_users)
    if missing_items > 0:
        print("Skipped %d rows because of missing items" % missing_items)
    return rows


def sort_by_user(rows):
    """Rows are sorted by item ids. Re-sort them by user ids instead
    """
    user_map = {}
    for row in rows:
        if row[0] in user_map:
            user_map[row[0]].append(row)
        else:
            user_map[row[0]] = [row]
    sorted_rows = []
    for user_id in sorted(list(user_map)):
        for row in user_map[user_id]:
            sorted_rows.append(row)
    return sorted_rows


def write_to_file(filename, rows):
    """Write the user item rating tuple to file
    """
    with open(filename, "w", newline='') as file:
        file.write("userId,itemId,rating\n")
        for row in rows:
            row = [str(i) for i in row]
            line = ",".join(row)
            file.write(line)
            file.write('\n')


if __name__ == '__main__':
    filename_train = "../data/datasets/netflix/netflix_train.txt"
    filename_test = "../data/datasets/netflix/netflix_test.txt"
    user_mapping = {}
    item_mapping = {}
    train_rows = sort_by_user(map_rows(filename_train, user_mapping, item_mapping))
    test_rows = sort_by_user(map_rows(filename_test, user_mapping, item_mapping, add_missing=False))
    filename_train_out = "../data/datasets/netflix/netflix_train_mapped.txt"
    filename_test_out = "../data/datasets/netflix/netflix_test_mapped.txt"
    write_to_file(filename_train_out, train_rows)
    write_to_file(filename_test_out, test_rows)
