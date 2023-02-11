
from AEMG.models import *
from AEMG.mg_utils import *
from AEMG.data_utils import *
np.set_printoptions(suppress=True)

config_fname = "config/cartpole_lqr.txt"

with open(config_fname, 'r') as f:
    config = eval(f.read())

mg_utils = MorseGraphUtils(config)

mg_out_utils = MorseGraphOutputProcessor(config)
attractor_nodes = mg_out_utils.attractor_nodes

ATTRACTOR_NODE = 0
traj_dataset = TrajectoryDataset(config)
ground_truth = []
predicted = []
for i in tqdm(range(len(traj_dataset))):
    ground_truth.append(mg_utils.system.achieved_goal(traj_dataset[i][-1]))
    enc = mg_utils.encode(traj_dataset[i][0])
    predicted.append(mg_out_utils.which_morse_set(enc) == ATTRACTOR_NODE)

print("Successful trials: ", sum(ground_truth))
print("Predicted successful trials: ", sum(predicted))

# Generate precision and recall
from sklearn.metrics import precision_recall_fscore_support
precision, recall, _, _ = precision_recall_fscore_support(ground_truth, predicted, average='binary')
print("Precision: ", precision)
print("Recall: ", recall)
