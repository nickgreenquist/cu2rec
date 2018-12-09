# cu2rec: CUDA Meets Recommender Systems

cu2rec is a Matrix Factorization library designed to accelerate training Recommender Systems models using GPUs in CUDA. It implements Parallel Stochastic Gradient Descent for training the matrix factorization model.

## Data
The input data should be a CSV file in the form of `userId,itemId,rating` and should have an header. If the user ids and the item ids are not sequential, run `python preprocessing/map_items.py <ratings_file>` to convert the user ids and item ids into sequential integers, starting with 1.

Once you have a mapped CSV, you can use `python preprocessing/split_to_test_train.py <mapped_file> <test_ratio>` to split the data into training and tests sets to use with `mf.cu`.

Alternatively, you can also use the datasets below:

### Movielens
1. Download movielens data [here](https://grouplens.org/datasets/movielens/) and save in `data` folder.
2. Run `python preprocessing/map_items.py <ratings_file>` to create a user-item mapped ratings file.
3. Run `python preprocessing/split_to_test_train.py <mapped_file> <test_ratio>` to split it into training and test files.

### Netflix
1. Download the Netflix dataset [here](https://drive.google.com/drive/folders/1ZxG4hVWqNGnlvPwx0T7lDwDq816GLXv-?usp=sharing) and place in under `data/datasets/netflix`.
2. Run `python preprocessing/map_netflix.py` to create the mapped training and test files.

## Compiling Code
1. SSH into Prince or `cuda2` using NYU credentials
2. `srun -t5:00:00 --mem=30000 --gres=gpu:1 --pty /bin/bash`
3. `module load cuda/9.2.88`
4. `cd matrix_factorization && make`

The makefile compiles for compute capability 5.2. If you have a GPU that does not support that, please change it to compile for your device's compute capability. The code has been tested for compute capability down to 3.5.

## Training
1. `make mf`
2. `bin/mf -c <config_file> <ratings_file_train> <ratings_file_test>`

## Experimental Results
### Netflix
1. Training Set MAE: 0.683889 RMSE: 0.873118
2. Test Set MAE: 0.776981 RMSE: 0.995426

## Getting recommendations for a user
1. Make sure you get the user data into the same ratings format as MovieLens.
2. `make predict`
3. `bin/predict -c <config_file> -i <trained_item_bias_file> -g <trained_global_bias_file> -q <trained_Q_file> <ratings_file>`

## Running Tests
1. `cd tests`
2. `make`
3. If you want to run all tests, `make run_all`
4. Otherwise, `bin/test_{}`

## Authors
- **[Nick Greenquist](https://nickgreenquist.github.io/)**
- **[Doruk Kilitcioglu](https://dorukkilitcioglu.github.io/)**
