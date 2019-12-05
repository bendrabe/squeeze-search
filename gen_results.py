import json
import os
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

out_dir_base = "trials/"
with open(out_dir_base+"results.txt", 'w') as res_file:
    res_file.write("TRIAL,INIT_LR,LR_DECAY_RATE,WEIGHT_DECAY,ACCURACY\n")
    for trial in os.listdir(out_dir_base):
        if trial == 'results.txt':
            continue
        out_dir = out_dir_base + trial
        ea = event_accumulator.EventAccumulator(out_dir+"/eval/")
        ea.Reload()
        acc = pd.DataFrame(ea.Scalars('accuracy'))['value'].max()
        print("loading json file {}".format(out_dir + "/hyperparams.txt"))
        with open(out_dir+"/hyperparams.txt", 'r') as f:
            hyper = json.load(f)
        if hyper['global_batch_size'] == 512:
            res_file.write("{},{},{},{},{}\n".format(trial, hyper['lr0'], hyper['lr_decay_rate'], hyper['weight_decay'], acc))
