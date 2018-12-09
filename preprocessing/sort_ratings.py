import argparse
import csv
import os
import random

def write_ratings(filename, ratings):
    with open(filename, "w", newline='') as file:
        file.write("userId,itemId,rating\n")
        for row in ratings:
            row = [str(i) for i in row]
            line = ",".join(row)
            file.write(line)
            file.write('\n')


def read_ratings(filename):
    rows = []
    # userId, itemId, rating, <other ignored columns>
    unique_users = set()
    unique_items = set()
    user_to_rating_count = {}
    with open(filename) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        row_num = 0
        for row in readCSV:
            if row_num > 0:
                userId = int(row[0])
                itemId = int(row[1])
                rating = float(row[2])

                rows.append([userId, itemId, rating])
            row_num += 1
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sorts ratings csv by userId, then itemId")
    parser.add_argument('file_ratings', type=str, help='the path to the ratings file to split')
    args = parser.parse_args()

    rows = read_ratings(args.file_ratings)

    # sort by userId, then itemId
    print("sorting ratings...")
    rows = sorted(rows, key=lambda e: (e[0], e[1]))
    print("done sorting ratings...")

    filepath, extension = os.path.splitext(args.file_ratings)

    write_ratings("{}_sorted{}".format(filepath, extension), rows)
