import matplotlib.pyplot as plt 
import pickle
import os

def plot_losses(config, key=None):
    with open(os.path.join(config['log_dir'], 'losses.pkl'), 'rb') as f:
        losses = pickle.load(f)
    
    if key is None:
        keys = losses['train_losses'].keys()
    else:
        keys = [key]
    
    for k in keys:
        fig = plt.figure(figsize=(8,8))
        plt.grid()
        plt.plot(losses['train_losses'][k], label='train')
        plt.plot(losses['test_losses'][k], label='val')
        # if k == "loss_total":
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.ylim(0, 0.1)
        plt.title(k)
        plt.legend(loc='best')
        plt.savefig(os.path.join(config['log_dir'], f'{k}.png'))