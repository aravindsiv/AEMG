import os
import argparse 
from tqdm import tqdm

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config',help='Base config file inside examples/config/',type=str,default='discrete_map.txt')
    parser.add_argument('--dir', help='Directory to save generated config files', type=str, default='tmp_config/')

    args = parser.parse_args()

    config_fname = f'config/{args.config}'

    with open(config_fname) as f:
        config = eval(f.read())
    
    if not os.path.exists(args.dir):
        os.makedirs(args.dir)

    uuid_generator = None
    try:
        import libpyDirtMP as prx
        uuid_generator = prx.generate_uuid()
    except:
        print("Using python uuid generator")
        import uuid
        uuid_generator = uuid.uuid4

    all_exps = []

    # Possible values for seed
    seeds = list(range(0, 10))
    # Possible values for the experiment
    exp_ids = [
        '1x1x1x', '1x100x1x'
    ]
    # Possible values for the number of layers
    num_layers = [1, 2]
    # Possible values for the data size (in k)
    data_size = [1, 10, 100]

    for exp in tqdm(exp_ids):
        for nl in num_layers:
            for ds in data_size:
                for seed in seeds:
                    row = {}
                    row['id'] = uuid_generator().hex
                    row['seed'] = seed
                    row['experiment'] = exp
                    row['num_layers'] = nl
                    row['data_size'] = ds
                
                    all_exps.append(row)

                    new_config = config.copy()
                    new_config['seed'] = seed
                    new_config['experiment'] = exp
                    new_config['num_layers'] = nl
                    new_config['data_dir'] = f"data/{new_config['system']}_{new_config['control']}{ds}k"
                    new_config['model_dir'] = f"tmp_models/{row['id']}/"
                    new_config['log_dir'] = f"tmp_logs/{row['id']}/"

                    with open(f'{args.dir}/{row["id"]}.txt', 'w') as f:
                        f.write(str(new_config))


    with open(f'all_exps.txt', 'w') as f:
        # Write as follows: <id>: <>,...
        for row in all_exps:
            f.write(f'{row["id"]}:')
            for key, value in row.items():
                if key == 'id':
                    continue
                f.write(f'{value},')
            f.write('\n')
    
    print("Generated all configs. Please save the all_exps.txt file.")

    '''
    # Possible values for the experiment

    exp_ids = [
        '111',
        '11e1x',
        '11e2x',
        '11e3x',
        'e1x11',
        'e2x11',
        'e3x11',
        '1e1x1',
        '1e2x1',
        '1e3x1',
        '100_111',
        '100_e1_11',
        '100_1e1x1',
        '100_11e1x',
        '100_011x111',
        '100_011_111',
        '100_011_11e1x',
        '100_011_11e2x',
        '100_011_e1x11',
        '100_011_e2x11',
        '100_011_111',
        '100_011_11e1x',
        '100_011_11e2x',
        '100_011_e1x11',
        '100_011_e2x11',
        '100_001_010_111',
        '100_001_010_11e1x',
        '100_001_010_11e2x',
        '100_001_010_e1x11',
        '100_001_010_e2x11',
        '100_001_010_1e1x1',
        '100_001_010_1e2x1',
        '100_001_010_1e1xe1x',
        '100_001_010_1e2xe2x',
        '100_001_010_e1x1e1x',
        '100_001_010_e2x1e2x',
        '100_001_010_e1xe1x1',
        '100_001_010_e1xe2x1',
        '100_001_110_001',
        '100_001_e1x10_001',
        '100_001_e2x10_001',
        '100_001_1e1x0_001',
        '100_001_1e2x0_001',
        '100_001_110_001_111',
        '100_001_e1x10_001_111',
        '100_001_e2x10_001_111',
        '100_001_1e1x0_001_111',
        '100_001_1e2x0_001_111',
        '100_001_110_001_11e1x',
        '100_001_110_001_11e2x',
        '100_001_110_001_1e1x1',
        '100_001_110_001_1e2x1',
        '100_001_110_001_e1x11',
        '100_001_110_001_e2x11',
        # Feel free to add more here
    ]
    # Possible values for the number of layers
    num_layers = [1, 2]
    # Data size in k for traing
    data_size = [1, 10, 100]



    # comment for few tests
    # exp_ids = ['1e2x1']
    # num_layers = [1]
    # seeds = [0]

    # Generate all possible combinations of the above

    for size in data_size:
        new_config_dir = f'config/{config["system"]}_{config["control"]}{str(size)}k'
        if not os.path.exists(new_config_dir):
            os.makedirs(new_config_dir)

        for index, exp_id in enumerate(exp_ids):
            new_dir = f'{new_config_dir}/{exp_id}'
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)

            for num_layer in num_layers:
                for seed in seeds:
                    # Create a new config dictionary
                    new_config = config.copy()

                    # Update the values
                    new_config['seed'] = seed
                    new_config['experiment'] = exp_id
                    new_config['num_layers'] = num_layer
                    new_config['data_dir'] = f"data/{new_config['system']}_{new_config['control']}{size}k"
                    new_config['model_dir'] = f"models/{new_config['system']}_{new_config['control']}{size}k/{exp_id}"
                    # new_config['data_dir'] = 'data/' + new_config['control'] + str(size) + 'k' 
                    # new_config['model_dir'] = 'models/' + new_config['control'] + str(size) + 'k/' + exp_id
                    
                    # Dump config to file 
                    # new_config_fname = os.path.join(new_config_dir, f'{exp_id}D{str(size)}kL{num_layer}S{seed}.txt')

                    new_config_fname = os.path.join(new_dir, f'{index}D{str(size)}kL{num_layer}S{seed}.txt')
                    with open(new_config_fname, 'w') as f:
                        f.write(str(new_config))
    '''