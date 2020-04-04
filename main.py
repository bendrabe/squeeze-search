import numpy as np
import os
import random

from train import Experiment

crop_space = ["squeeze", "resnet"]
std_space = [False, True]
mixup_space = [False, True]
lr0_space = [0.03, 0.04, 0.06]
lrdp_space = [0.75, 1.0]
wd_space = [0.0002, 0.0004]
wepochs_space = [0, 4]

combined_space = [(i,j,k,l,m,n,o)
                  for o in wepochs_space
                  for n in wd_space
                  for m in lrdp_space
                  for l in lr0_space
                  for k in mixup_space
                  for j in std_space
                  for i in crop_space]

model_dir_base = 'trials/'
if not os.path.exists(model_dir_base):
    os.makedirs(model_dir_base)

trial = 0
while os.path.exists(model_dir_base + str(trial)):
    trial += 1
    # sometimes we go up, sometimes we go down
    if trial >= len(combined_space) or trial < 0:
        print("All gridpts processed, exiting...")
        exit()
model_dir = model_dir_base + str(trial) + '/'
os.makedirs(model_dir)

crop, std, mixup, lr0, lrdp, wd, wepochs = combined_space[trial]

'''
exp = Experiment(model_dir=model_dir,
                 crop=crop,
                 std=std,
                 mixup=mixup,
                 lr0=lr0,
                 lr_decay_rate=lrdp,
                 weight_decay=wd,
                 warmup_epochs=wepochs)
'''
exp = Experiment(model_dir=model_dir, crop="squeeze", mixup=True)
exp.log_hyperparams()
exp.execute()
