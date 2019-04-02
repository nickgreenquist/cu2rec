import argparse
import numpy as np
import os


def save_as_npy(filename_in, filename_out):
    arr = np.genfromtxt(filename_in, delimiter=',')
    np.save(filename_out, arr)


def get_filename_out(filename_in):
    filepath, _ = os.path.splitext(filename_in)
    return "{0}.npy".format(filepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Converts float matrices in CSV form to numpy arrays")
    parser.add_argument('files', type=str, nargs='+', help="the files to convert")
    args = parser.parse_args()

    for filename in args.files:
        filename_out = get_filename_out(filename)
        save_as_npy(filename, filename_out)
