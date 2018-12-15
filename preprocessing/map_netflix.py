"""Maps the Netflix ratings to a format readable by mf.cu.

Make sure you download the Netflix ratings from the link in the README.
"""

from map_items import sort_by_user, map_rows, write_to_file


def get_netflix_info(row):
    user_id = int(row[0])
    item_id = int(row[1])
    rating = float(row[3])  # There are two spaces before the ratings
    return user_id, item_id, rating


if __name__ == '__main__':
    filename_train = "../data/datasets/netflix/netflix_train.txt"
    filename_test = "../data/datasets/netflix/netflix_test.txt"
    user_mapping = {}
    item_mapping = {}
    train_rows = sort_by_user(map_rows(filename_train, user_mapping, item_mapping,
                                       delimiter=' ', has_header=False, get_info=get_netflix_info))
    test_rows = sort_by_user(map_rows(filename_test, user_mapping, item_mapping, delimiter=' ',
                                      has_header=False, get_info=get_netflix_info, add_missing=False))
    filename_train_out = "../data/datasets/netflix/ratings_mapped_train.csv"
    filename_test_out = "../data/datasets/netflix/ratings_mapped_test.csv"
    write_to_file(filename_train_out, train_rows)
    write_to_file(filename_test_out, test_rows)
