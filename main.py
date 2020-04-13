import numpy as np
import os
import random

from train import Experiment

MODE = 'random'
MODEL_DIR_BASE = 'trials/'

def get_next_trialnum(model_dir_base, max_trialnum=None):
    trial = 0
    while os.path.exists(model_dir_base + str(trial)):
        trial += 1
        if max_trialnum and trial > max_trialnum:
            print("All trials complete, exiting...")
            exit()
    return trial

# draws uniformly in log domain, then exponentiates
# see http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf
def draw_exponentially(space_min, space_max):
    log_val = np.log(space_min) + (np.log(space_max) - np.log(space_min))*np.random.rand()
    return np.exp(log_val)

# hparams common to both grid and random
crop_space = ["squeeze", "resnet"]
bool_space = [False, True]
wepochs_space = [0, 4]
base_lr = 0.04
base_lrdp = 1.0
base_wd = 0.0002
base_mixup = 0.2

if MODE == 'grid':
    mixup_space = [None, 0.2]
    lr0_space = [0.04, 0.07]
    lrdp_space = [0.75, 1.0]
    wd_space = [0.0002, 0.0004]

    combined_space = [(i,j,k,l,m,n,o)
                      for o in wepochs_space
                      for n in wd_space
                      for m in lrdp_space
                      for l in lr0_space
                      for k in mixup_space
                      for j in bool_space
                      for i in crop_space]
    # need to process all gridpts in interval [0, len(combined_space)-1]
    trial = get_next_trialnum(MODEL_DIR_BASE, len(combined_space)-1)
    crop, std, mixup, lr0, lrdp, wd, wepochs = combined_space[trial]
else:
    # trial num used for convenience only
    trial = get_next_trialnum(MODEL_DIR_BASE)
    crop = random.choice(crop_space)
    std = random.choice(bool_space)
    mixup = None
    if random.choice(bool_space):
        mixup = draw_exponentially(base_mixup/2, base_mixup*2)
    lr0 = draw_exponentially(base_lr/2, base_lr*2)
    lrdp = draw_exponentially(base_lrdp/2, base_lrdp*2)
    wd = draw_exponentially(base_wd/2, base_wd*2)
    wepochs = random.choice(wepochs_space)

if not os.path.exists(MODEL_DIR_BASE):
    os.makedirs(MODEL_DIR_BASE)
model_dir = MODEL_DIR_BASE + str(trial) + '/'
os.makedirs(model_dir)

if mixup is not None:
    num_epochs = 90
else:
    num_epochs = 68

exp = Experiment(num_epochs=num_epochs,
                 model_dir=model_dir,
                 data_dir='/data/imagenet-tfrecord/',
                 crop=crop,
                 std=std,
                 mixup=mixup,
                 lr0=lr0,
                 lr_decay_rate=lrdp,
                 weight_decay=wd,
                 warmup_epochs=wepochs)
exp.log_hyperparams()
exp.execute()
