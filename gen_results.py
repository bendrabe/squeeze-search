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
    res_file.write("TRIAL,CROP,STD,MIXUP,INIT_LR,LR_DECAY_RATE,WEIGHT_DECAY,WARMUP_EPOCHS,VAL_ACC,TEST_ACC\n")
    for trial in os.listdir(out_dir_base):
        if trial == 'results.txt':
            continue

        out_dir = out_dir_base + trial
        if not os.path.isdir(out_dir+"/eval_val/") or not os.path.isdir(out_dir+"/eval_test/"):
            print("trial {} is missing eval data, skipping...".format(out_dir))
            continue
        print("adding trial {} to results.txt".format(out_dir))

        ea_val = event_accumulator.EventAccumulator(out_dir+"/eval_val/")
        ea_val.Reload()
        acc_val = pd.DataFrame(ea_val.Scalars('accuracy'))['value'].max()

        ea_test = event_accumulator.EventAccumulator(out_dir+"/eval_test/")
        ea_test.Reload()
        acc_test = pd.DataFrame(ea_test.Scalars('accuracy'))['value'].max()

        with open(out_dir+"/hyperparams.txt", 'r') as f:
            hyper = json.load(f)
        res_file.write("{},{},{},{},{},{},{},{},{},{}\n".format(trial, hyper['crop'], hyper['std'], hyper['mixup'], hyper['lr0'], hyper['lr_decay_rate'], hyper['weight_decay'], hyper['warmup_epochs'], acc_val, acc_test))
