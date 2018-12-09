"""Sorts the data CSV using userId first and itemId second.
"""

import argparse
import csv
import os

from map_items import write_to_file


def read_ratings(filename):
    rows = []
    # userId, itemId, rating, <other ignored columns>
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

    write_to_file("{}_sorted{}".format(filepath, extension), rows)
