import json
import os
import pandas as pd
import sys
from tensorboard.backend.event_processing import event_accumulator

if not len(sys.argv) == 2 or not os.path.isdir(sys.argv[1]):
	print("takes directory holding numbered experiment trials as argument, exiting...")
	exit()

out_dir_base = sys.argv[1] + "/"
with open(out_dir_base+"results.txt", 'w') as res_file:
    res_file.write("TRIAL,GLOBAL_BS,INIT_LR,LR_DECAY_RATE,WEIGHT_DECAY,WARMUP_EPOCHS,ACCURACY\n")
    for trial in os.listdir(out_dir_base):
        if trial == 'results.txt':
            continue
        out_dir = out_dir_base + trial
        if not os.path.isdir(out_dir+"/eval/"):
            print("trial {} is missing eval data, skipping...".format(out_dir))
            continue
        print("adding trial {} to results.txt".format(out_dir))
        ea = event_accumulator.EventAccumulator(out_dir+"/eval/")
        ea.Reload()
        acc = pd.DataFrame(ea.Scalars('accuracy'))['value'].max()
        with open(out_dir+"/hyperparams.txt", 'r') as f:
            hyper = json.load(f)
        if 'lr_decay_power' in hyper:
            res_file.write("{},{},{},{},{},0,{}\n".format(trial, hyper['global_batch_size'], hyper['lr0'], hyper['lr_decay_power'], hyper['weight_decay'], acc))
        elif 'warmup_epochs' in hyper:
            res_file.write("{},{},{},{},{},{},{}\n".format(trial, hyper['global_batch_size'], hyper['lr0'], hyper['lr_decay_rate'], hyper['weight_decay'], hyper['warmup_epochs'], acc))
        else:
            res_file.write("{},{},{},{},{},0,{}\n".format(trial, hyper['global_batch_size'], hyper['lr0'], hyper['lr_decay_rate'], hyper['weight_decay'], acc))
