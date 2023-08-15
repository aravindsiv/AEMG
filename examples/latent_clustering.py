from AEMG.data_utils import *
from AEMG.dynamics_utils import *
from AEMG.mg_utils import *

from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import precision_recall_fscore_support

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment',help='Directory of results inside output/',type=str,default='pendulum_lqr1k')
    parser.add_argument('--id', type=str, default="")
    parser.add_argument('--print', action='store_true')
    parser.add_argument('--output_dir',type=str, default="")
    parser.add_argument('--labels_file',help="Success/Failure labels inside output/",type=str, default='')
    parser.add_argument('--use_final', action='store_true')
    parser.add_argument('--cluster_threshold', help='Minimum # of dataset points that need to belong to a cluster', type=int, default=0)
    args = parser.parse_args()

    if args.id == "":
        config_fnames = os.listdir(os.path.join("output/",args.experiment))
    else:
        config_fnames = [args.id]
    
    print("Assuming all configs have the same dataset.")
    config_fname = os.path.join("output",args.experiment,config_fnames[0], "config.txt")
    with open(config_fname, 'r') as f:
        config = eval(f.read())
    assert config['low_dims'] == 2, "Only 2D systems supported"

    attractors = None
    if config['system'] == 'pendulum':
        attractors = np.array([[-2.1, 0.0], [0.0, 0.0], [2.1, 0.0]])
    elif config['system'] == 'bistable':
        attractors = np.array([[-1.39]+[0.0]*9, [1.39]+[0.0]*9])
    elif config['system'] == 'cartpole':
        attractors = np.array([[0.0, 0.0, 0.0, 0.0],
                            [1.0, np.pi, 0.0, 0.0],
                            [-1.0, -np.pi, 0.0, 0.0]])
    else:
        if args.labels_file == '':
            print("No labels file provided")
            exit(0)
        else:
            trajectories = TrajectoryDataset(config, os.path.join("output", args.labels_file))
            if args.use_final:
                attractors = trajectories.get_successful_final_conditions()
            else:
                attractors = trajectories.get_successful_initial_conditions()
    
    for fname in tqdm(config_fnames):

        config_fname = os.path.join("output",args.experiment,fname, "config.txt")

        with open(config_fname, 'r') as f:
            config = eval(f.read())
        
        dynamics = DynamicsUtils(config)

        try:
            mg_out_utils = MorseGraphOutputProcessor(config)
        except FileNotFoundError or ValueError:
            exit(0)

        latent_final_conditions = []
        if not args.use_final:
            for i in range(len(attractors)):
                z_curr = dynamics.encode(attractors[i])
                for j in range(12//config['step']):
                    z_next = dynamics.f(z_curr)
                    z_curr = z_next
                latent_final_conditions.append(z_curr)
        else:
            for i in range(len(attractors)):
                latent_final_conditions.append(dynamics.encode(attractors[i]))
        
        latent_final_conditions = np.array(latent_final_conditions)
        
        k_choices = range(2,11)
        scores = []
        # Perform clustering for each k and use elbow method to choose k
        for k in k_choices:
            # Cluster attractors
            kmeans = KMeans(n_clusters=k, random_state=0).fit(latent_final_conditions)
            labels = kmeans.labels_
            # Compute silhouette score
            score = silhouette_score(latent_final_conditions, labels)
            scores.append(score)
        
        best_score = np.argmax(scores)
        print("Best k = ", k_choices[best_score], ", score = ", scores[best_score])
        # Do clustering again with best k
        kmeans = KMeans(n_clusters=k_choices[best_score], random_state=0).fit(latent_final_conditions)
        labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_

        cluster_threshold = 0.01 * int(args.cluster_threshold) * latent_final_conditions.shape[0]
        
        # Plot attractors
        plt.figure(figsize=(8,8))
        plt.xlim(-1,1)
        plt.ylim(-1,1)

        attractors_of_interest = set()
        for i in range(cluster_centers.shape[0]):
            # Check # of points in each cluster
            print("Cluster {}:".format(i))
            if np.sum(labels == i) < cluster_threshold: continue
            gt_att_enc = cluster_centers[i]
            attractor_dists = []
            for j in range(mg_out_utils.get_num_attractors()):
                attractor_tiles = mg_out_utils.get_corner_points_of_attractor(mg_out_utils.attractor_nodes[j])
                min_dist = np.inf
                for k in range(attractor_tiles.shape[0]):
                    cp_low = attractor_tiles[k, :config['low_dims']]
                    cp_high = attractor_tiles[k, config['low_dims']:]
                    tile_center = (cp_low + cp_high) / 2.0
                    min_dist = min(min_dist, np.linalg.norm(tile_center - gt_att_enc))
                    plt.scatter(tile_center[0], tile_center[1], c='black', s=100./attractor_tiles.shape[0], marker='x')
                attractor_dists.append(min_dist)
            print("Distances to MG attractors: ",attractor_dists)
            attractors_of_interest.add(mg_out_utils.attractor_nodes[np.argmin(attractor_dists)])
       
        plt.scatter(latent_final_conditions[:,0], latent_final_conditions[:,1], c=labels, cmap='viridis', marker='*')
        plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='black', s=200, alpha=0.5)
        plt.title("Attractors")
        plt.savefig(os.path.join(config['output_dir'], "clustering.png"))

        ground_truth = []
        predicted = []

        for i in tqdm(range(len(trajectories))):
            ground_truth.append(trajectories.get_label(i))
            enc = dynamics.encode(trajectories[i][0])
            morse_set = mg_out_utils.which_morse_set(enc)
            print("Morse set: ",morse_set)
            if morse_set in attractors_of_interest:
                predicted.append(1)
            else:
                predicted.append(0)
        
        precision, recall, f1, _ = precision_recall_fscore_support(ground_truth, predicted, average='binary')
        print("Precision: ", precision)
        print("Recall: ", recall)

        # Write attractors of interest to file
        with open(os.path.join(config['output_dir'], "attractors_of_interest.txt"), 'w') as f:
            f.write(str(list(attractors_of_interest))+ "\n")
            f.write(str(precision) + "\n")
            f.write(str(recall) + "\n")
            