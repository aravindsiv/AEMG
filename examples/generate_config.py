import os 

config_fname = 'config/discrete_map.txt'

with open(config_fname) as f:
    config = eval(f.read())

# Possible values for seed
seeds = list(range(0, 100))
# Possible values for the experiment
exp_ids = [
    'All',
    'Enc_L1*All',    
    'Enc_L1*Dyn_L2L3*All',
    'Enc_L1*Dyn_L2*All',
    'Enc_L1*Dyn_L3*Enc_L2*All',
    # Feel free to add more here
]
# Possible values for the number of layers
num_layers = [1, 2]

new_config_dir = 'test_config'
if not os.path.exists(new_config_dir):
    os.makedirs(new_config_dir)

# Generate all possible combinations of the above
for seed in seeds:
    for exp_id in exp_ids:
        for num_layer in num_layers:
            # Create a new config dictionary
            new_config = config.copy()

            # Update the values
            new_config['seed'] = seed
            new_config['experiment'] = exp_id
            new_config['num_layers'] = num_layer

            # Dump config to file 
            new_config_fname = os.path.join(new_config_dir, f'seed_{seed}_exp_{exp_id}_num_layers_{num_layer}.txt')
            with open(new_config_fname, 'w') as f:
                f.write(str(new_config))