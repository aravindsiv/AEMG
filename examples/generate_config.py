import os
import argparse 
import uuid
from tqdm import tqdm

def get_exp_ids(num_design):
    exp_ids = [
        '500x1x1x0x',
        '1x100x1x0x',
        '1x1x1x0x',
        '500x1x1x1x',
        '1x100x1x1x',
        '1x1x1x1x',
        '500x1x1x0.1x',
        '1x100x1x0.1x',
        '1x1x1x0.1x',
        '500x1x1x0.01x',
        '1x100x1x0.01x',
        '1x1x1x0.01x',
        '5x1x1x',
        '10x1x1x',
        '50x1x1x',
        '100x1x1x',
        '1000x1x1x',
        '1x5x1x',
        '1x10x1x',
        '1x50x1x',
        '1x100x1x',
        '1x500x1x',
        '1x1000x1x',
        '1x1x5x',
        '1x1x10x',
        '1x1x50x',
        '1x1x100x',
        '1x1x500x',
        '1x1x1000x',
        '1x0x0x_1x1x1x',
        '1x0x0x_10x_1x1x',
        '1x0x0x_1x10x1x',
        '1x0x0x_1x1x10x',
        '1x0x0x_100x_1x1x',
        '1x0x0x_1x100x1x',
        '1x0x0x_1x1x100x',
        '1x0x0x_10x1x10x',
        '1x0x0x_100x1x100x',
        '1x0x0x_0x1x1x_1x1x1x',
        '1x0x0x_0x1x1x_1x1x10x',
        '1x0x0x_0x1x1x_1x1x100x',
        '1x0x0x_0x1x1x_10x1x1x',
        '1x0x0x_0x1x1x_100x1x1x',
        '1x0x0x_0x1x1x_1x10x1x',
        '1x0x0x_0x1x1x_1x100x1x',
        '1x0x0x_0x1x1x_10x1x10x',
        '1x0x0x_0x1x1x_100x1x100x',
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
        '1x0x0x_0x0x1x_1x1x0x_0x0x1x_1x1x1x',
        '1x0x0x_0x0x1x_10x1x0x_0x0x1x_1x1x1x',
        '1x0x0x_0x0x1x_100x1x0x_0x0x1x_1x1x1x',
        '1x0x0x_0x0x1x_1x10x0x_0x0x1x_1x1x1x',
        '1x0x0x_0x0x1x_1x100x0x_0x0x1x_1x1x1x',
        '1x0x0x_0x0x1x_1x1x0x_0x0x1x_1x1x10x',
        '1x0x0x_0x0x1x_1x1x0x_0x0x1x_1x1x100x',
        '1x0x0x_0x0x1x_1x1x0x_0x0x1x_1x10x1x',
        '1x0x0x_0x0x1x_1x1x0x_0x0x1x_1x100x1x',
        '1x0x0x_0x0x1x_1x1x0x_0x0x1x_10x1x1x',
        '1x0x0x_0x0x1x_1x1x0x_0x0x1x_100x1x1x',
    ]
    return exp_ids[0:num_design]

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
    save_folder = os.path.join(os.getcwd(),f"tmp_sh_{args.name}{args.out_extra}")
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
    save_folder = os.path.join(os.getcwd(),f"tmp_sh_{args.name}{args.out_extra}")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    lines = exp_cluster_array()

    shell_name = f"{save_folder}/{args.shell}.sh"
    with open(shell_name, "w") as file:
        file.write(lines[0])
        file.write(f"#SBATCH --array=0-{dir_counter}\n")

        for line in lines[1::]:
            
            if line[0:11] == "search_dir=":
                write_path = f"\nsearch_dir={path_config}$SLURM_ARRAY_TASK_ID/\n"
                file.write(write_path)
                continue

            file.write(line)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config',help='Base config file inside examples/config/',type=str,default='discrete_map.txt')
    parser.add_argument('--dir', help='Directory to save generated config files', type=str, default='tmp_config')
    parser.add_argument('--name', help='Name of the experiment', type=str,required=True)
    parser.add_argument('--max_jobs',help='Split into multiple files',type=int,default=100)
    parser.add_argument('--shell',help='Generate shell script to send job',type=str,default="")
    parser.add_argument('--seed', help='Select the number of experiments', type=int, default=10)
    parser.add_argument('--num_layers', help='Select the number of layers', type=int, default=1)
    parser.add_argument('--data_size', help='Select data size (accept seq of numbers)', action='store', type=int, nargs='*', default=[1])
    parser.add_argument('--num_design', help='Select the number of designs', type=int, default=3)
    parser.add_argument('--num_steps', help='Select the number of steps', type=int, default=1)
    parser.add_argument('--out_extra',help='Add extra name to output',type=str,default="")

    args = parser.parse_args()

    config_fname = f'config/{args.config}'

    with open(config_fname) as f:
        config = eval(f.read())
    
    args.dir = args.dir[0:-1] + args.out_extra + '/'
    if not os.path.exists(args.dir):
        os.makedirs(args.dir)

    uuid_generator = uuid.uuid4

    all_exps = []

    # Possible values for seed
    seeds = list(range(0, args.seed))
    # Possible values for the experiment
    exp_ids = get_exp_ids(args.num_design)
    # Possible values for the number of layers
    num_layers = list(range(1, args.num_layers + 1))
    # Possible values for the data size (in k)
    data_size = args.data_size #[10**i for i in range(args.data_size)]

    # Possible values for steps
    steps = [1, 3, 6, 12][0:args.num_steps]

    # # Output all files here
    # output = f'{os.getcwd()}/output/{args.name}'
    # if not os.path.exists(output):
    #     os.makedirs(output)

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

                        # update the values of the base config file
                        new_config = config.copy()
                        new_config['seed'] = seed
                        new_config['experiment'] = exp
                        new_config['num_layers'] = nl
                        new_config['data_dir'] = f"data/{new_config['system']}_{new_config['control']}{ds}k"
                        system_control_dir_temp = f"{new_config['system']}_{new_config['control']}{ds}k{args.out_extra}"
                        output_temp = f"output/{system_control_dir_temp}/{row['id']}/"
                        new_config['output_dir'] = output_temp
                        new_config['model_dir'] = f"{output_temp}model/"
                        new_config['log_dir'] = f"{output_temp}logs/"
                        new_config["step"] = step

                        counter += 1
                        if counter % args.max_jobs == 0 and counter != 1: 
                            dir_counter += 1
                            # if args.shell !="": generate_shell(args, f'{args.dir}{dir_counter}', dir_counter)
                        
                        # save temp config to run experiments
                        temp_dir_exp = f'{args.dir}{dir_counter}'
                        if not os.path.exists(temp_dir_exp):
                            os.makedirs(temp_dir_exp)
                        
                        with open(f'{temp_dir_exp}/{row["id"]}.txt', 'w') as f:
                            f.write(str(new_config))

                        # save config for future references
                        if not os.path.exists(output_temp):
                            os.makedirs(output_temp)
                        with open(f'{output_temp}/config.txt', 'w') as f:
                            f.write(str(new_config))
    print(dir_counter)
    if args.shell !="": generate_job_array(args, f'{args.dir}', dir_counter)


    
    with open(f'output/{system_control_dir_temp}/all_exps.txt', 'w') as f:
        # Write as follows: <id>: <>,...
        for row in all_exps:
            f.write(f'{row["id"]}:')
            for key, value in row.items():
                if key == 'id':
                    continue
                f.write(f'{value},')
            f.write('\n')
    
    print("Generated all configs. Please save the all_exps.txt file.")