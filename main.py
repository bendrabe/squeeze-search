import numpy as np
import os
import random

from train import Experiment

_NUM_TRAIN_IMAGES=1281167

def trunc_norm(loc, scale, a, b):
    r = np.random.normal(loc, scale)
    while r < a or r > b:
        r = np.random.normal(loc, scale)
    return r

BASE_WD = 0.0002

batch_space = [256, 512]
batch_size = random.choice(batch_space)

lr0_space_min = 0.01
lr0_space_max = 1.0
log_lr0 = np.log(lr0_space_min) + (np.log(lr0_space_max) - np.log(lr0_space_min))*np.random.rand()
lr0 = np.exp(log_lr0)

lrf_space_min = (lr0 / 100) / 2
lrf_space_max = (lr0 / 100) * 2
log_lrf = np.log(lrf_space_min) + (np.log(lrf_space_max) - np.log(lrf_space_min))*np.random.rand()
lrf = np.exp(log_lrf)

steps_per_epoch = ((_NUM_TRAIN_IMAGES - 1 ) // batch_size) + 1

b = (-1 / (60*steps_per_epoch))*np.log( lrf / lr0 )

wd_space_min = BASE_WD / 2
wd_space_max = BASE_WD * 2
log_wd = np.log(wd_space_min) + (np.log(wd_space_max) - np.log(wd_space_min))*trunc_norm(0.5, 0.25, 0, 1)
wd = np.exp(log_wd)

model_dir_base = 'trials/'
if not os.path.exists(model_dir_base):
    os.makedirs(model_dir_base)
trial = 0
while os.path.exists(model_dir_base + str(trial)):
    trial += 1
model_dir = model_dir_base + str(trial) + '/'
os.makedirs(model_dir)

exp = Experiment(model_dir=model_dir,
                 global_batch_size=batch_size,
                 lr0=lr0,
                 lr_decay_rate=b,
                 weight_decay=wd)
exp.log_hyperparams()
exp.execute()
