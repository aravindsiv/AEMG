python3 generate_config.py

k="000"
w="k"
for i in 1
do
    # ik="$i$k"
    # echo $ik
    # echo "data/bistable$i$w/"
    python3 get_data_map.py --num_trajs "$i$k" --save_dir "data/bistable$i$w/"
done


search_dir=test_config/
yourfilenames=`ls $(pwd)/test_config/*.txt`

for eachfile in $yourfilenames
do
    echo $(basename $eachfile)
    python3 train.py --config_dir "$search_dir" --config "$(basename $eachfile)"
    python3 get_MG_RoA.py --config_dir "$search_dir" --config "$(basename $eachfile)"
done