import numpy as np
import os
import random

from train import Experiment

NUM_TRAIN_IMAGES = 1281167
BATCH_SIZE = 512
NUM_EPOCHS = 68

lr0_mult = 6
lr0_space_min = 0.02
lr0_space_max = 0.1

lrdp_space_min = 0.75
lrdp_space_max = 1.5

wd_mult = 4
wd_space_min = 0.0002
wd_space_max = 0.0004

lr0_space = list(np.geomspace(lr0_space_min, lr0_space_max, lr0_mult))
lrdp_space = list(np.arange(lrdp_space_min, lrdp_space_max, 0.25))
wd_space = list(np.geomspace(wd_space_min, wd_space_max, wd_mult))
wepochs_space = [0, 2, 4, 8]

combined_space = [(i,j,k,l)
                  for l in wepochs_space
                  for k in wd_space
                  for j in lrdp_space
                  for i in lr0_space]

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

lr0, lrdp, wd, wepochs = combined_space[trial]

exp = Experiment(model_dir=model_dir,
                 lr0=lr0,
                 lr_decay_rate=lrdp,
                 warmup_epochs=wepochs,
                 weight_decay=wd)
exp.log_hyperparams()
exp.execute()
