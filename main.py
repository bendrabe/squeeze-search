import numpy as np
import os
import random

from train import Experiment

NUM_TRAIN_IMAGES = 1281167
BATCH_SIZE = 512
NUM_EPOCHS = 70
LATE_LR_EPOCH = 4*NUM_EPOCHS//5

lr0_mult = 8
wd_mult = 5
lr0_space_min = 0.02
lr0_space_max = 0.2
wd_space_min = 0.00015
wd_space_max = 0.0004
lr0_space = list(np.geomspace(lr0_space_min, lr0_space_max, lr0_mult))
wd_space = list(np.geomspace(wd_space_min, wd_space_max, wd_mult))
combined_space = [(i,j) for i in list(lr0_space) for j in list(wd_space)]

model_dir_base = 'trials/'
if not os.path.exists(model_dir_base):
    os.makedirs(model_dir_base)
trial = 0
while os.path.exists(model_dir_base + str(trial)):
    trial += 1
    if trial == len(combined_space):
        print("All gridpts processed, exiting...")
        exit()
model_dir = model_dir_base + str(trial) + '/'
os.makedirs(model_dir)

lr0, wd = combined_space[trial]
lrf = lr0 / 100

steps_per_epoch = ((NUM_TRAIN_IMAGES - 1 ) // BATCH_SIZE) + 1
b = (-1 / (LATE_LR_EPOCH*steps_per_epoch))*np.log( lrf / lr0 )

exp = Experiment(model_dir=model_dir,
                 lr0=lr0,
                 lr_decay_rate=b,
                 weight_decay=wd)
exp.log_hyperparams()
exp.execute()
