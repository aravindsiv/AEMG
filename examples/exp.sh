search_dir=test_config/
yourfilenames=`ls $(pwd)/test_config/*.txt`

for eachfile in $yourfilenames
do
    echo $(basename $eachfile)
    python train.py --config_dir "$search_dir" --config "$(basename $eachfile)"
    python get_MG_RoA.py --config_dir "$search_dir" --config "$(basename $eachfile)"
done