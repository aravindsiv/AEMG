import os
import argparse 
from tqdm import tqdm

def get_exp_ids():
    exp_ids = ['1x1x1x', '1x100x1x']
    exp_ids = [
        '1x1x1x',
        '1x1x5x',
        '1x1x10x',
        '1x1x50x',
        '1x1x100x',
        '1x1x500x',
        '1x1x1000x',
        '5x1x1x',
        '10x1x1x',
        '50x1x1x',
        '100x1x1x',
        '500x1x1x',
        '1000x1x1x',
        '1x5x1x',
        '1x10x1x',
        '1x50x1x',
        '1x100x1x',
        '1x500x1x',
        '1x1000x1x',
        # '1x0x0x_1x1x1x',
        # '1x0x0x_10x_1x1x',
        # '1x0x0x_1x10x1x',
        # '1x0x0x_1x1x10x',
        # '1x0x0x_100x_1x1x',
        # '1x0x0x_1x100x1x',
        # '1x0x0x_1x1x100x',
        # '1x0x0x_10x1x10x',
        # '1x0x0x_100x1x100x',
        # '1x0x0x_0x1x1x_1x1x1x',
        # '1x0x0x_0x1x1x_1x1x10x',
        # '1x0x0x_0x1x1x_1x1x100x',
        # '1x0x0x_0x1x1x_10x1x1x',
        # '1x0x0x_0x1x1x_100x1x1x',
        # '1x0x0x_0x1x1x_1x10x1x',
        # '1x0x0x_0x1x1x_1x100x1x',
        # '1x0x0x_0x1x1x_10x1x10x',
        # '1x0x0x_0x1x1x_100x1x100x',
        '1x0x0x_0x0x1x_0x1x0x_1x1x1x',
        '1x0x0x_0x0x1x_0x1x0x_1x1x10x',
        '1x0x0x_0x0x1x_0x1x0x_1x1x100x',
        '1x0x0x_0x0x1x_0x1x0x_10x1x1x',
        '1x0x0x_0x0x1x_0x1x0x_100x1x1x',
        '1x0x0x_0x0x1x_0x1x0x_1x10x1x',
        '1x0x0x_0x0x1x_0x1x0x_1x100x1x',
        '1x0x0x_0x0x1x_0x1x0x_1x10x10x',
        '1x0x0x_0x0x1x_0x1x0x_1x100x100x',
        '1x0x0x_0x0x1x_0x1x0x_10x1x10x',
        '1x0x0x_0x0x1x_0x1x0x_100x1x100x',
        '1x0x0x_0x0x1x_0x1x0x_10x10x1x',
        '1x0x0x_0x0x1x_0x1x0x_10x100x1x',
        '1x0x0x_0x0x1x_1x1x0x_0x0x1x',
        '1x0x0x_0x0x1x_10x1x0x_0x0x1x',
        '1x0x0x_0x0x1x_100x1x0x_0x0x1x',
        '1x0x0x_0x0x1x_1x10x0x_0x0x1x',
        '1x0x0x_0x0x1x_1x100x0x_0x0x1x',
        # '1x0x0x_0x0x1x_1x1x0x_0x0x1x_1x1x1x',
        # '1x0x0x_0x0x1x_10x1x0x_0x0x1x_1x1x1x',
        # '1x0x0x_0x0x1x_100x1x0x_0x0x1x_1x1x1x',
        # '1x0x0x_0x0x1x_1x10x0x_0x0x1x_1x1x1x',
        # '1x0x0x_0x0x1x_1x100x0x_0x0x1x_1x1x1x',
        # '1x0x0x_0x0x1x_1x1x0x_0x0x1x_1x1x10x',
        # '1x0x0x_0x0x1x_1x1x0x_0x0x1x_1x1x100x',
        # '1x0x0x_0x0x1x_1x1x0x_0x0x1x_1x10x1x',
        # '1x0x0x_0x0x1x_1x1x0x_0x0x1x_1x100x1x',
        # '1x0x0x_0x0x1x_1x1x0x_0x0x1x_10x1x1x',
        # '1x0x0x_0x0x1x_1x1x0x_0x0x1x_100x1x1x',
    ]
    return exp_ids

def exp_cluster(name_sh = "/run/aemg.sh"):
    name_sh = os.getcwd() + name_sh

    with open(name_sh, "r") as reader:
        halfs = reader.read().split("# split_here #")
    return halfs[0], halfs[1]

def exp_cluster_array(name_sh = "/run/aemg_array.sh"):
    name_sh = os.getcwd() + name_sh

    with open(name_sh, "r") as reader:
        lines = reader.readlines()
        lines.pop(1)
    return lines

def generate_shell(args, path_config, dir_counter):
    save_folder = os.path.join(os.getcwd(),f"tmp_sh_{args.name}")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    halfs_0, halfs_1 = exp_cluster()


    shell_name = f"{save_folder}/{args.shell}{dir_counter}.sh"
    with open(shell_name, "w") as file:
        file.write(halfs_0)

        write_path = f"\nsearch_dir={path_config}/\n"
        file.write(write_path)

        file.write(halfs_1)

def generate_job_array(args, path_config, dir_counter):
    save_folder = os.path.join(os.getcwd(),f"tmp_sh_{args.name}")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    lines = exp_cluster_array()

    shell_name = f"{save_folder}/{args.shell}{dir_counter}.sh"
    with open(shell_name, "w") as file:
        file.write(lines[0])
        file.write(f"#SBATCH --array=1-{dir_counter}")

        for line in lines[1::]:
            file.write(line)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config',help='Base config file inside examples/config/',type=str,default='discrete_map.txt')
    parser.add_argument('--dir', help='Directory to save generated config files', type=str, default='tmp_config/')
    parser.add_argument('--name', help='Name of the experiment', type=str,required=True)
    parser.add_argument('--max_jobs',help='Split into multiple files',type=int,default=1000)
    parser.add_argument('--shell',help='Generate shell script to send job',type=str,default="")

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
    seeds = list(range(0, 1000))
    # Possible values for the experiment
    
    exp_ids = get_exp_ids()
    # Possible values for the number of layers
    num_layers = [1, 2]
    # Possible values for the data size (in k)
    data_size = [1, 10, 100]

    # Possible values for steps
    steps = [1, 3, 6]

    counter = 0
    dir_counter = 0
    for exp in tqdm(exp_ids):
        for ds in data_size:
            for seed in seeds:
                for nl in num_layers:
                    for step in steps:
                        # row is used to save a uuid file
                        row = {}
                        row['id'] = uuid_generator().hex
                        row['seed'] = seed
                        row['experiment'] = exp
                        row['num_layers'] = nl
                        row['data_size'] = f"{config['system']}_{config['control']}{ds}k"
                        row["step"] = step
                    
                        all_exps.append(row)

                        # new_config is a new config file generated by base config file
                        new_config = config.copy()
                        new_config['seed'] = seed
                        new_config['experiment'] = exp
                        new_config['num_layers'] = nl
                        new_config['data_dir'] = f"data/{new_config['system']}_{new_config['control']}{ds}k"
                        new_config['model_dir'] = f"tmp_models/{row['id']}/"
                        new_config['log_dir'] = f"tmp_logs/{row['id']}/"
                        new_config["step"] = step

                        counter += 1
                        if counter % args.max_jobs == 0: 
                            dir_counter += 1
                            if args.shell !="": generate_shell(args, f'{args.dir}{dir_counter}', dir_counter)
                        if not os.path.exists(f'{args.dir}{dir_counter}'):
                            os.makedirs(f'{args.dir}{dir_counter}')

                        # with open(f'{args.dir}/{row["id"]}.txt', 'w') as f:
                        with open(f'{args.dir}{dir_counter}/{row["id"]}.txt', 'w') as f:
                            f.write(str(new_config))

    if args.shell !="": generate_shell(args, f'{args.dir}{dir_counter}', dir_counter)

    output = os.getcwd() + "/tmp_all_exps"
    if not os.path.exists(output):
        os.makedirs(output)
    
    with open(f'{output}/{args.name}_all_exps.txt', 'w') as f:
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