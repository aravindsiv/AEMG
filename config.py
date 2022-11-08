import os


mode = 'noctrl'

steps = 20
data_file = f"pendulum_{mode}_0.1_{steps}step_1M.txt"
ROOT_PATH = f'root_{mode}'
train = True
warmup = 0
high_dims = 4
low_dims = 2

# class training:
epochs = 100
lr = 1e-3
batch_size = 1024