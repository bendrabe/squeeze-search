import numpy as np
import os
import random

from train import Experiment

def trunc_norm(loc, scale, a, b):
    r = np.random.normal(loc, scale)
    while r < a or r > b:
        r = np.random.normal(loc, scale)
    return r

BASE_LR = 0.04
BASE_WD = 0.0002
BASE_LRDP = 1.0

batch_space = [128, 256, 512, 1024]
batch_size = random.choice(batch_space)

mean_lr = BASE_LR * batch_size / 512
lr_space_min = mean_lr / 2
lr_space_max = mean_lr * 2
log_lr = np.log(lr_space_min) + (np.log(lr_space_max) - np.log(lr_space_min))*trunc_norm(0.5, 0.25, 0, 1)
lr = np.exp(log_lr)

wd_space_min = BASE_WD / 2
wd_space_max = BASE_WD * 2
log_wd = np.log(wd_space_min) + (np.log(wd_space_max) - np.log(wd_space_min))*trunc_norm(0.5, 0.25, 0, 1)
wd = np.exp(log_wd)

lrdp_space_min = BASE_LRDP / 2
lrdp_space_max = BASE_LRDP * 2
log_lrdp = np.log(lrdp_space_min) + (np.log(lrdp_space_max) - np.log(lrdp_space_min))*trunc_norm(0.5, 0.25, 0, 1)
lrdp = np.exp(log_lrdp)

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
                 lr0=lr,
                 weight_decay=wd,
                 lr_decay_power=lrdp)
exp.log_hyperparams()
exp.execute()
