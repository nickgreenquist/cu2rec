"""Maps items in a ratings CSV file to sequential user and item ids.

By default, assumes the first column is user id, second is item id,
and third is rating. If different, a different function can be passed into
map_rows as the get_info parameter.
"""

import argparse
import csv
import os


def get_sequential_info(row):
    # userId, itemId, rating, <other ignored columns>
    user_id = int(row[0])
    item_id = int(row[1])
    rating = float(row[2])
    return user_id, item_id, rating


def map_rows(filename, user_mapping, item_mapping, delimiter=',', has_header=True,
             get_info=get_sequential_info, add_missing=True):
    """Maps the rows of the file with user_mapping and item_mapping.

    If add_missing is True, it will automatically add the missing user and
    item ids to the mappings. They are mapped sequentially starting from 1.
    get_info is a function callback that returns the userid-itemid-rating tuple
    from a row.
    """
    rows = []
    missing_users = 0
    missing_items = 0
    with open(filename) as csvfile:
        csv_rows = csv.reader(csvfile, delimiter=delimiter)
        if has_header:
            next(csv_rows, None)
        for row in csv_rows:
            user_id, item_id, rating = get_info(row)

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
    """Sort the rows by user
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


def process_file(filename_in, filename_out):
    user_mapping = {}
    item_mapping = {}
    rows = sort_by_user(map_rows(filename_in, user_mapping, item_mapping))
    write_to_file(filename_out, rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Maps user and item ids to sequential ids, starting from 1")
    parser.add_argument('file_ratings', type=str, help='the path to the ratings file to map')
    args = parser.parse_args()

    filepath, extension = os.path.splitext(args.file_ratings)
    filename_out = "{}_mapped{}".format(filepath, extension)
    process_file(args.file_ratings, filename_out)
