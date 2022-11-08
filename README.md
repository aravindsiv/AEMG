Use the config.py file to change some of the important training variables

### Collect data
1. Set the mode in the config.py file to either `"lqr"` or `"noctrl"`
2. run `python get_data.py`

### Train model

#### LQR
1.  set the mode `mode="lqr"`
2.  set `warmup=20`
3.  set `epochs=120`


#### no control
1.  set the mode `mode="noctrl"`
2.  set `warmup=0`
3.  set `epochs=100`

Run `python train_dynamics_ae.py`

This should create a folder either `root_lqr` or `root_noctrl` which contains the models and the resulting visualizations.
You can use the models directly from inside these folders.
