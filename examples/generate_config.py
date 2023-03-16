import os 

if __name__ == "__main__":
    config_fname = 'config/discrete_map.txt'

    with open(config_fname) as f:
        config = eval(f.read())

    # Possible values for seed
    seeds = list(range(0, 100))
    # Possible values for the experiment
    exp_ids = [
        '1_1_1',
        '1_1_10',
        '1_1_100',
        '1_1_1000',
        '10_1_1',
        '100_1_1',
        '1000_1_1',
        '1_10_1',
        '1_100_1',
        '1_1000_1',
        '1_0_0*1_1_1',
        '1_0_0*10_1_1',
        '1_0_0*1_10_1',
        '1_0_0*1_1_10',
        '1_0_0*0_1_1*1_1_1',
        '1_0_0*0_1_1*1_1_10',
        '1_0_0*0_1_1*1_1_100',
        '1_0_0*0_1_1*10_1_1',
        '1_0_0*0_1_1*100_1_1',
        '1_0_0*0_1_0*1_1_1',
        '1_0_0*0_1_0*1_1_10',
        '1_0_0*0_1_0*1_1_100',
        '1_0_0*0_1_0*10_1_1',
        '1_0_0*0_1_0*100_1_1',
        '1_0_0*0_1_0*1_10_1',
        '1_0_0*0_1_0*1_100_1',
        '1_0_0*0_0_1*0_1_0*1_1_1',
        '1_0_0*0_0_1*0_1_0*1_1_10',
        '1_0_0*0_0_1*0_1_0*1_1_100',
        '1_0_0*0_0_1*0_1_0*10_1_1',
        '1_0_0*0_0_1*0_1_0*100_1_1',
        '1_0_0*0_0_1*0_1_0*1_10_1',
        '1_0_0*0_0_1*0_1_0*1_100_1',
        '1_0_0*0_0_1*0_1_0*1_10_10',
        '1_0_0*0_0_1*0_1_0*1_100_100',
        '1_0_0*0_0_1*0_1_0*10_1_10',
        '1_0_0*0_0_1*0_1_0*100_1_100',
        '1_0_0*0_0_1*0_1_0*10_10_1',
        '1_0_0*0_0_1*0_1_0*10_100_1',
        '1_0_0*0_0_1*1_1_0*0_0_1',
        '1_0_0*0_0_1*10_1_0*0_0_1',
        '1_0_0*0_0_1*100_1_0*0_0_1',
        '1_0_0*0_0_1*1_10_0*0_0_1',
        '1_0_0*0_0_1*1_100_0*0_0_1',
        '1_0_0*0_0_1*1_1_0*0_0_1*1_1_1',
        '1_0_0*0_0_1*10_1_0*0_0_1*1_1_1',
        '1_0_0*0_0_1*100_1_0*0_0_1*1_1_1',
        '1_0_0*0_0_1*1_10_0*0_0_1*1_1_1',
        '1_0_0*0_0_1*1_100_0*0_0_1*1_1_1',
        '1_0_0*0_0_1*1_1_0*0_0_1*1_1_10',
        '1_0_0*0_0_1*1_1_0*0_0_1*1_1_100',
        '1_0_0*0_0_1*1_1_0*0_0_1*1_10_1',
        '1_0_0*0_0_1*1_1_0*0_0_1*1_100_1',
        '1_0_0*0_0_1*1_1_0*0_0_1*10_1_1',
        '1_0_0*0_0_1*1_1_0*0_0_1*100_1_1'
        # Feel free to add more here
    ]
    # Possible values for the number of layers
    num_layers = [1, 2]
    # Data size in k for traing
    data_size = [1, 10, 100]

    new_config_dir = 'test_config'
    if not os.path.exists(new_config_dir):
        os.makedirs(new_config_dir)

    # comment for few tests
    # exp_ids = ['1_0_0*0_0_1*1_1_0*0_0_1*100_1_1']
    # num_layers = [1]
    # seeds = [0,1,2]

    # Generate all possible combinations of the above
    
    for exp_id in exp_ids:
        for size in data_size:
            for num_layer in num_layers:
                for seed in seeds:
                    # Create a new config dictionary
                    new_config = config.copy()

                    # Update the values
                    new_config['seed'] = seed
                    new_config['experiment'] = exp_id
                    new_config['num_layers'] = num_layer
                    new_config['data_dir'] = 'data/' + new_config['control'] + str(size) + 'k'

                    # Dump config to file 
                    new_config_fname = os.path.join(new_config_dir, f'{exp_id}D{str(size)}kL{num_layer}S{seed}.txt')
                    with open(new_config_fname, 'w') as f:
                        f.write(str(new_config))