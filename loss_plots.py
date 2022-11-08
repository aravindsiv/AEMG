from matplotlib import pyplot as plt
import pickle
import os

PATH = 'lqr'
if not os.path.exists(PATH):
    os.makedirs(PATH)

with open('losses_warmup_lqr.pkl', 'rb') as f:
    losses = pickle.load(f)

for k, v in losses['train'].items():
    fig = plt.figure(figsize=(8,8))
    plt.plot(losses['train'][k], label='train')
    plt.plot(losses['val'][k], label='val')
    # if k == "loss_total":
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 0.003)
    plt.title(k)
    plt.legend()
    plt.savefig(f'{PATH}/{k}.png')
    # plt.show()