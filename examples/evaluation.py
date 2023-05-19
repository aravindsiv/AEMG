
from AEMG.mg_utils import *
from AEMG.data_utils import *
from AEMG.dynamics_utils import *
np.set_printoptions(suppress=True)

config_fname = "config/pendulum_lqr_1K.txt"

with open(config_fname, 'r') as f:
    config = eval(f.read())

dynamics = DynamicsUtils(config)

mg_out_utils = MorseGraphOutputProcessor(config)
attractor_nodes = mg_out_utils.attractor_nodes

GT_NUM_ATTRACTORS = 3
print("# of attractors for the ground truth system: ", GT_NUM_ATTRACTORS)
print("# of attractors for the learned system: ", mg_out_utils.get_num_attractors())

# Now, we will attempt to find an injective mapping from 
# the attractors of the ground truth system to the learned system
# Shown below for a single attractor
GT_ATTRACTOR = dynamics.system.transform(np.array([0.0, 0.0]))
gt_att_enc = dynamics.encode(GT_ATTRACTOR)

min_dist = np.inf 
identified_attractor = -1
for i in range(mg_out_utils.get_num_attractors()):
    attractor_tiles = mg_out_utils.get_corner_points_of_attractor(i)
    for j in range(attractor_tiles.shape[0]):
        cp_low = attractor_tiles[j, :config['low_dims']]
        cp_high = attractor_tiles[j, config['low_dims']:]
        tile_center = (cp_low + cp_high) / 2.0
        if np.linalg.norm(tile_center - gt_att_enc) < min_dist:
            min_dist = np.linalg.norm(tile_center - gt_att_enc)
            identified_attractor = i

print("Identified attractor: ", identified_attractor, " with distance: ", min_dist)

def is_gt_roa(pt):
    pt_low_dim = dynamics.system.inverse_transform(pt)
    if np.linalg.norm(pt_low_dim) < 0.1:
        return True
    return False

traj_dataset = TrajectoryDataset(config)
ground_truth = []
predicted = []
for i in tqdm(range(len(traj_dataset))):
    # Check if the last point of the trajectory is inside the attractor
    ground_truth.append(is_gt_roa(traj_dataset[i][-1]))
    # Encode the first point of the trajectory
    enc = dynamics.encode(traj_dataset[i][0])
    predicted.append(mg_out_utils.which_morse_set(enc) == identified_attractor)

print("GT RoA points: ", sum(ground_truth))
print("Predicted RoA points: ", sum(predicted))

# Generate precision and recall
from sklearn.metrics import precision_recall_fscore_support
precision, recall, _, _ = precision_recall_fscore_support(ground_truth, predicted, average='binary')
print("Precision: ", precision)
print("Recall: ", recall)
