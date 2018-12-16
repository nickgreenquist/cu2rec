"""Creates a config file readable by mf.cu

Since JSON support doesn't seem to be on the cards right now, this is a quick way of
generating config files.
"""

import argparse


def save_to_file(args):
    with open(args.filename, 'w') as fp:
        # 0 100 10 0.0001 42 0.2 0.1 0.1 0.1
        cfg = '0 {:d} {:d} {:f} {:d} {:f} {:f} {:f} {:f}'.format(
            args.num_iterations, args.num_factors, args.learning_rate, args.seed,
            args.p_reg, args.q_reg, args.user_bias_reg, args.item_bias_reg,
            # TODO: add these to config
            # args.num_threads, args.patience, args.learning_rate_decay
        )
        fp.write(cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Creates a config file to be used with mf.cu")
    parser.add_argument('filename', type=str, help='the name of the config file to be created')
    parser.add_argument('-n', '--num_iterations', type=int, default=1000, help="the total number of iterations")
    parser.add_argument('-f', '--num_factors', type=int, default=100, help="the number of factors to use")
    parser.add_argument('-l', '--learning_rate', type=float, default=0.01, help="the learning rate")
    parser.add_argument('-s', '--seed', type=int, default=42, help="the seed for the random number generator")
    parser.add_argument('-p', '--p_reg', type=float, default=0.02, help="the regularization parameter for the user matrix")
    parser.add_argument('-q', '--q_reg', type=float, default=0.02, help="the regularization parameter for the item matrix")
    parser.add_argument('-u', '--user_bias_reg', type=float, default=0.02, help="the regularization parameter for the user biases")
    parser.add_argument('-i', '--item_bias_reg', type=float, default=0.02, help="the regularization parameter for the item biases")
    parser.add_argument('-t', '--num_threads', type=int, default=32, help="the number of threads in a block, must be from 2^0 to 2^9")
    parser.add_argument('-a', '--patience', type=int, default=2, help="he number of times loss can stay constant or increase before triggering a learning rate decay")
    parser.add_argument('-d', '--learning_rate_decay', type=float, default=0.2, help="the amount of decay of learning rate after patience is exceeded")
    args = parser.parse_args()
    save_to_file(args)
