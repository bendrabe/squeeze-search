import numpy as np
import os
import random

from train import Experiment

NUM_TRAIN_IMAGES = 1281167
BATCH_SIZE = 512
NUM_EPOCHS = 70
LATE_LR_EPOCH = 4*NUM_EPOCHS//5
WEIGHT_DECAY = 0.0004

lr0_mult = 8
lr0_space_min = 0.02
lr0_space_max = 0.2
lrf_space_min = lr0_space_min / 100
lrf_space_max = lr0_space_max / 100
lr0_space = list(np.geomspace(lr0_space_min, lr0_space_max, lr0_mult))
lrf_space = list(np.arange(lrf_space_min, lrf_space_max, 0.0001))
combined_space = [(i,j) for i in list(lr0_space) for j in list(lrf_space)]

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

lr0, lrf = combined_space[trial]

steps_per_epoch = ((NUM_TRAIN_IMAGES - 1 ) // BATCH_SIZE) + 1
b = (-1 / (LATE_LR_EPOCH*steps_per_epoch))*np.log( lrf / lr0 )

exp = Experiment(model_dir=model_dir,
                 lr0=lr0,
                 lr_decay_rate=b,
                 weight_decay=WEIGHT_DECAY)
exp.log_hyperparams()
exp.execute()
