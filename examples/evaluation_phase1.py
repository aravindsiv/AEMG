from AEMG.mg_utils import *
from AEMG.data_utils import *
from AEMG.dynamics_utils import *
np.set_printoptions(suppress=True)

import os
import argparse
from tqdm import tqdm
from collections import defaultdict

def generate_experiment_id(design, num_layers, step):
    return design + "&" + str(num_layers) + str(step)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment',help='Directory of results inside output/',type=str,default='pendulum_lqr1k')
    parser.add_argument('--desired_num_attractors',help='Desired number of attractors',type=int,default=2)

    results_dir = "output/" + parser.parse_args().experiment
    GT_NUM_ATTRACTORS = parser.parse_args().desired_num_attractors

    attractors_success = {}
    more_attaractors_success = {}

    for dir in tqdm(os.listdir(results_dir)):
        if dir.endswith('.txt'): continue
        config_fname = os.path.join(results_dir, dir, "config.txt")

        with open(config_fname) as f:
            config = eval(f.read())
        
        try:
            mg_out_utils = MorseGraphOutputProcessor(config)
        except FileNotFoundError: 
            continue
        except ValueError:
            print("ValueError from: ", config_fname)
            continue
        attractor_nodes = mg_out_utils.attractor_nodes

        id = generate_experiment_id(config['experiment'], config['num_layers'], config['step'])
        if id not in attractors_success:
            attractors_success[id] = []
            more_attaractors_success[id] = []
        
        attractors_success[id].append(mg_out_utils.get_num_attractors() == GT_NUM_ATTRACTORS)
        more_attaractors_success[id].append(mg_out_utils.get_num_attractors() >= GT_NUM_ATTRACTORS)
    
    with open(results_dir + "_phase1.txt","w") as f:
        for k in attractors_success.keys():
            f.write(k + "," + str(np.mean(attractors_success[k])) + ","
                    + str(np.mean(more_attaractors_success[k])) + "\n")

if __name__ == "__main__":
    main()