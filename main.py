import numpy as np
import os
import random

from train import Experiment

NUM_TRAIN_IMAGES = 1281167
BATCH_SIZE = 512
NUM_EPOCHS = 68

lr0_space_min = 0.02
lr0_space_max = 0.1
log_lr0 = np.log(lr0_space_min) + (np.log(lr0_space_max) - np.log(lr0_space_min))*np.random.rand()
lr0 = np.exp(log_lr0)

lrdp_space_min = 0.75
lrdp_space_max = 1.5
log_lrdp = np.log(lrdp_space_min) + (np.log(lrdp_space_max) - np.log(lrdp_space_min))*np.random.rand()
lrdp = np.exp(log_lrdp)

wd_space_min = 0.0002
wd_space_max = 0.0004
log_wd = np.log(wd_space_min) + (np.log(wd_space_max) - np.log(wd_space_min))*np.random.rand()
wd = np.exp(log_wd)

wepochs_space = [0, 2, 4, 8]
wepochs = random.choice(wepochs_space)

model_dir_base = 'trials/'
if not os.path.exists(model_dir_base):
    os.makedirs(model_dir_base)

trial = 0
while os.path.exists(model_dir_base + str(trial)):
    trial += 1
'''
    # sometimes we go up, sometimes we go down
    if trial >= len(combined_space) or trial < 0:
        print("All gridpts processed, exiting...")
        exit()
'''
model_dir = model_dir_base + str(trial) + '/'
os.makedirs(model_dir)

exp = Experiment(model_dir=model_dir,
                 lr0=lr0,
                 lr_decay_rate=lrdp,
                 warmup_epochs=wepochs,
                 weight_decay=wd)
exp.log_hyperparams()
exp.execute()
