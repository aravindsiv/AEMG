import os

mode = 'lqr'
WARMUP = 20
EPOCHS = 120

# mode = 'noctrl'
# WARMUP = 0
# EPOCHS = 100

theta_thresh = 0.5

steps = 20
data_file = f"pendulum_{mode}_0.1_{steps}step_1M.txt"
ROOT_PATH = f'root_{mode}'
train = True
warmup = WARMUP
high_dims = 4
low_dims = 2

# class training:
epochs = EPOCHS
lr = 1e-3
batch_size = 1024
