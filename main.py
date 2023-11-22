'''
Created by Luca La Fisca
ISIA Lab, Faculty of Engineering University of Mons, Mons (Belgium)
luca.lafisca@umons.ac.be
Source: La Fisca et al, "Enhancing OSA Assessment with Explainable AI", EMBC 2023
Copyright (C) 2023 - UMons
This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.
This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
This library cannot be used for commercial use without the agreement of the
author (Luca La Fisca).
'''
from collections import Counter

import zarr
from fastai.tabular.all import *
from fastai.data.all import *
from fastai.vision.gan import *
from fastai import *
from tsai.all import *
from torch import nn
import numpy as np
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import torch.nn.functional as F
from model import stagerNetAAE, stagerNetCritic
from utils import LossAttrMetric, GetLatentSpace, norm_batch, UnfreezeFcCrit, \
                SwitchAttribute, distrib_regul_regression, hist_lab, plot_results

# Load the config file
config_file = 'config.json'
with open(config_file, 'r') as file:
    config = json.load(file)

# Set the device on which you want to train the model
device = torch.device(config['device'])
torch.cuda.set_device(device)

lab_area = torch.Tensor(np.load(f'{config["labels_path"]}/area_db.npy'))[:,None]
lab_arousal = torch.Tensor(np.load(f'{config["labels_path"]}/arousal_db.npy'))[:,None]
lab_duration = torch.Tensor(np.load(f'{config["labels_path"]}/duration_db.npy'))[:,None]

# Define the labels
# 1) discrete labels
lab_area = torch.Tensor(np.load(f'{config["labels_path"]}/area_db.npy'))[:,None]
lab_arousal = torch.Tensor(np.load(f'{config["labels_path"]}/arousal_db.npy'))[:,None]
lab_duration = torch.Tensor(np.load(f'{config["labels_path"]}/duration_db.npy'))[:,None]
lab_all = torch.Tensor(4*lab_area + 2*lab_arousal + lab_duration)
lab_discrete = torch.hstack((lab_area,lab_duration,lab_arousal))
# 2) switch to match the desired encoding
tmp = copy(lab_all)
lab_all[tmp==3] = 4
lab_all[tmp==4] = 3
# 3) 3-level labels ("low", "medium", "high")
lab3 = deepcopy(lab_all)
lab3[:] = 0
lab3[lab_all>1] = 1
lab3[lab_all>5] = 2
# 4) 4-level labels ("all metrics at low level", "1 metrics at high level", "2 metrics at high level", "all metrics at high level")
lab4 = deepcopy(lab_all)
lab4[lab_all>0] = 1
lab4[lab_all>3] = 2
lab4[lab_all==7] = 3
# 5) normalize the label values
lab_norm_area = torch.Tensor(np.load(f'{config["labels_path"]}/norm_area_db.npy')).unsqueeze(-1)
lab_norm_duration = torch.Tensor(np.load(f'{config["labels_path"]}/norm_duration_db.npy')).unsqueeze(-1)
lab_norm = torch.hstack((lab_norm_area,lab_norm_duration,lab_arousal))
#normalize the binary arousal value with respect to the std of area and duration labels 
lab_arousal_tmp = torch.Tensor([-1 if x==0 else 1 for x in lab_arousal]).unsqueeze(-1)
lab_norm_arousal = lab_arousal_tmp * (lab_norm_area.std() + lab_norm_duration.std()) / 2
lab_gather = torch.hstack((lab_norm_area,lab_norm_duration,lab_norm_arousal))
lab_gather = lab_gather.mean(dim=1).unsqueeze(-1) # mean of all metrics
# 6) Gather all the labels in a list in right order
label_stack = torch.hstack((lab_gather, lab_area, lab_duration, lab_arousal, lab3, lab4))

# Define dls
if config['load_dls']:
    dls = torch.load(config['dls_path']) # should be a .pkl file

else:
    # Read your data (.zarr file)
    path = Path(config['data_path'])
    X = zarr.open(path, mode='r')
    t = torch.Tensor(X)
    print('data properly read')

    # Define splitter
    n_train_samples = round(len(t)*config['trainset_part'])
    n_total_samples = len(t)
    splits = (L(range(n_train_samples), use_list=True),
              L(np.arange(n_train_samples, n_total_samples), use_list=True))
    splitter = IndexSplitter(splits[1])

    getters = [ItemGetter(0), ItemGetter(1)]
    dblock = DataBlock(blocks=(TSTensorBlock,TSTensorBlock),
                       getters=getters,
                       splitter=splitter,
                       batch_tfms=norm_batch())
    src = itemify(t.to('cpu'),label_stack.to('cpu'))
    dls = dblock.dataloaders(src, bs=config['bs'], val_bs=config['val_bs'], drop_last=True)

    torch.save(dls, config['dls_path'])

    # free memory space
    del X
    time.sleep(.2)
    torch.cuda.empty_cache()
    print('memory flushed')

dls = dls.to(device)
print('dls:')
print(dls.one_batch())


### Train the AutoEncoder part ###
acc_factor = config['acc_factor']
latent_dim = config['latent_dim']
model = stagerNetAAE(latent_dim=latent_dim,acc_factor=acc_factor)
model = model.to(device)
if config['train_ae']:
    metrics = [rmse]
    learn = Learner(dls, model, loss_func = model.ae_loss_func, metrics=metrics, opt_func=ranger)
    learning_rate = learn.lr_find()
    learn.fit_flat_cos(n_epoch=config['n_epoch'], lr=learning_rate.valley,
                        cbs=[
                            GradientAccumulation(n_acc=dls.bs*acc_factor),
                            TrackerCallback(),
                            SaveModelCallback(fname=config['ae_filename']),
                            EarlyStoppingCallback(min_delta=1e-4,patience=config['patience'])])

state_dict = torch.load(f'models/{config["ae_filename"]}.pth') # load the best weights


### Train the Classifier part ###
classif_filename = config['classif_filename']
model.load_state_dict(state_dict, strict=False)
#define the metrics to show
metrics = [LossAttrMetric("gather_loss"), LossAttrMetric("simple_loss"),
           LossAttrMetric("area_loss"), LossAttrMetric("duration_loss"),
           LossAttrMetric("arousal_loss"), LossAttrMetric("ord_loss")]
#freeze the discriminator weights
for name, param in model.named_parameters():
    if "fc_crit" in name:
        param.requires_grad_(False)

if config['train_classif_discrete']:
    #define the losses to montitor
    monitor_loss = ['area_loss','duration_loss','arousal_loss']
    #set the learning rates
    learning_rates = [1e-3,5e-4,2e-4]
    # Start curriculum learning
    total_cycles = config['nb_of_metrics']
    for i in range(total_cycles):
        curr_filename = str(classif_filename)+'_level'+str(i+1)
        model.level = i+1
        met = metrics[1:i+3] + metrics[-1:]
        learn = Learner(dls, model, loss_func=model.classif_loss_func,
                       metrics=met, opt_func=ranger)

        learn.fit_flat_cos(config['n_epoch'], lr=learning_rates[i],
                            cbs=[
                                GradientAccumulation(n_acc=dls.bs*acc_factor),
                                TrackerCallback(monitor=monitor_loss[i]),
                                SaveModelCallback(fname=curr_filename,monitor=monitor_loss[i]),
                                EarlyStoppingCallback(min_delta=1e-4,patience=config['patience'],monitor=monitor_loss[i]),
                                SwitchAttribute(attribute_name='global_loss', switch_every=5)
                                ])
        learn.load(curr_filename)
        model.load_state_dict(learn.model.state_dict())

state_dict = torch.load(f'models/{classif_filename}_level3.pth') # load the best weights
model.load_state_dict(state_dict, strict=False)
if config['train_regress']:
    model.level = 0
    model.dropout_rate = .1
    learn = Learner(dls, model, loss_func=model.classif_loss_func,
                   metrics=metrics, opt_func=ranger)
    learn.fit_flat_cos(config['n_epoch'], lr=1e-3,
                            cbs=[
                                GradientAccumulation(n_acc=dls.bs*acc_factor),
                                TrackerCallback(monitor='gather_loss'),
                                SaveModelCallback(fname=classif_filename, monitor='gather_loss'),
                                EarlyStoppingCallback(min_delta=1e-4,patience=config['patience'],monitor='gather_loss'),
                                SwitchAttribute(attribute_name='global_loss', switch_every=5)])

    np.save('results/'+str(classif_filename)+'_losses.npy', learn.recorder.losses)
    np.save('results/'+str(classif_filename)+'_values.npy', learn.recorder.values)

state_dict = torch.load(f'models/{config["classif_filename"]}.pth') # load the best weights


### Train the Adversarial part ###
model.load_state_dict(state_dict, strict=False)
adv_filename = config['aae_filename']
if config['train_aae']:
    metrics = [LossAttrMetric("classif_loss"), LossAttrMetric("recons_loss"),
               LossAttrMetric("adv_loss")]
    learn = Learner(dls, model, loss_func=model.aae_loss_func,
                   metrics=metrics, opt_func=ranger)

    learn.fit_flat_cos(config['n_epoch'], lr=1e-3,
                            cbs=[
                                GradientAccumulation(n_acc=dls.bs*acc_factor),
                                TrackerCallback(monitor='classif_loss'),
                                SaveModelCallback(fname=adv_filename, monitor='classif_loss'),
                                EarlyStoppingCallback(min_delta=1e-4,patience=config['patience'],monitor='classif_loss'),
                                UnfreezeFcCrit(switch_every=2),
                                SwitchAttribute(attribute_name='global_loss', switch_every=5)])

state_dict = torch.load(f'models/{adv_filename}.pth') # load the best weights


### Extract the latent space ###
result_filename = config['result_filename']
model.load_state_dict(state_dict, strict=False)
learn = Learner(dls,model,loss_func=model.aae_loss_func)
if config['load_latent_space']:
    new_zi = torch.load(f'data/z_{result_filename}.pt')
    print(f'latent space loaded with shape {new_zi.shape}')
else:
    learn.zi_valid = torch.tensor([]).to(device)
    learn.get_preds(ds_idx=0,cbs=[GetLatentSpace(cycle_len=1)])
    new_zi = learn.zi_valid
    learn.zi_valid = torch.tensor([]).to(device)
    learn.get_preds(ds_idx=1,cbs=[GetLatentSpace(cycle_len=1)])
    new_zi = torch.vstack((new_zi,learn.zi_valid))

print("new_zi shape: "+str(new_zi.shape))
torch.save(new_zi,f'data/z_{result_filename}.pt')

### Display the latent space ###
plot_results(new_zi.to(device),lab_gather,learn,result_filename)