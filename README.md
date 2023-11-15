# xAAEnet
Scoring the Severity of Sleep Disorders With Explainable AI.

This repository provides the open-source codes and supplementary materials related to the publications:   
- [La Fisca *et al*, "Explainable AI for EEG Biomarkers Identification in Obstructive Sleep Apnea Severity Scoring Task", NER 2023.](https://ieeexplore.ieee.org/abstract/document/10123795)
- [La Fisca *et al*, "Enhancing OSA Assessment with Explainable AI", EMBC 2023.](https://orbi.umons.ac.be/handle/20.500.12907/46450)

## Requirements
The library versions used here are:
- python=3.8
- pytorch=1.11.0
- cudatoolkit=11.3
- fastai=2.7.9
- fastcore=1.5.24
- matplotlib=3.5.2
- numpy=1.22.3
- scikit-learn=1.1.0
- scipy=1.8.0
- seaborn=0.11.2
- torchmetrics=0.7.3
- tsai=0.3.1
- zarr=2.12.0

All the required packages could be directly installed within your conda environment by using the file requirements.txt through:
```
conda create -n <environment-name> --file requirements.txt
```

/!\ Your input data should be stored as [.zarr file](https://zarr.readthedocs.io/en/stable/tutorial.html) /!\

## Tutorial
### 1. Set the config file
Modify the [config.json file](https://github.com/numediart/xAAEnet/config.json).

Define device and data splitting
```
"device" 				: "cuda:3",
"trainset_part" 		: 0.75,
```
Define what to load and what to train
```
"load_dls" 				: false,
"load_latent_space" 	: false,
"train_ae" 				: true,
"train_classif_discrete": true,
"train_regress" 		: true,
"train_aae" 			: true,
```
Define the required paths
```
"data_path" 			: "/home/JennebauffeC/pytorchVAE/fastAI/data/X_large.zarr",
"dls_path" 				: "./data/dls.pkl",
"labels_path" 			: "./data",
```
Define the model filenames
```
"ae_filename" 			: "xaaenet_ae",
"classif_filename" 		: "xaaenet_classif",
"aae_filename" 			: "xaaenet_aae",
"result_filename" 		: "result",
```
Define the required parameters
```
"nb_of_labels"			: 6, # number of labels on which you will perform a classification/regression (here: lab_gather, lab_area, lab_duration, lab_arousal, lab3, lab4)
"nb_of_metrics"			: 3, # number of metrics of reference (here: desaturation area, apnea duration, and arousal events)
"bs" 					: 16, # training batch size
"val_bs"				: 32, # validation batch size
"latent_dim"  			: 128,	# the dimension of the latent vector on which you focus your analysis
"acc_factor" 			: 16, # number of batchs that will be gathered together before updating the weights of your neural network
"n_epoch" 				: 200, # number of epochs of each training phase
"patience" 				: 25 # number of epochs without improvement for triggering early stopping
```

### 2. Define your labels
Here, all the metrics values are stored as npy files in /data folder
```
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
```

### 3. Define your data loader
Use the zarr file as input and the gathered labels as output
```
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
src = itemify(t,label_stack)
dls = dblock.dataloaders(src, bs=config['bs'], val_bs=config['val_bs'], drop_last=True)

torch.save(dls, config['dls_path'])
```
### 4. Define your model
Modify the architecture of stagerNetAAE to use the model you want.

In __init__, define your layers
```
def __init__(self, channels: int=23, timestamps: int=3001,
                acc_factor: int=8, dropout_rate: float=0.5, level: int=0,
                latent_dim: int=128, gan_depth: int=3, k_pool_size: int=13
            ):
    super(stagerNetAAE, self).__init__()
    
    self.channels = channels #number of input channels (spatial)
    self.timestamps = timestamps #number of input timestamps (temporal)
    self.acc_factor = acc_factor
    self.latent_dim = latent_dim #embed_dim
    self.k_pool_size = k_pool_size #embed_dim
    self.dropout_rate = dropout_rate
    self.gan_depth = gan_depth
    self.gen_train = True
    self.level = level
    self.global_loss = False
    
    #=============Encoder=============#
    self.conv1 = nn.Conv2d(1, self.channels, (1, self.channels), stride=(1, 1))
    self.conv2 = nn.Conv2d(1, 16, (self.timestamps//60,1), stride=(1,1))
    self.conv3 = nn.Conv2d(16, 16, (self.timestamps//60,1), stride=(1,1))
    self.pool = nn.MaxPool2d(kernel_size=(self.k_pool_size,1), return_indices=True) 
    self.batchnorm1 = nn.BatchNorm2d(16)
    self.batchnorm2 = nn.BatchNorm2d(16)
    self.bn_lin = nn.BatchNorm1d(num_features=self.latent_dim)

    self.fc_input_size = self._get_fc_input_size(self.timestamps)
    self.fc_z = nn.Linear(self.fc_input_size, self.latent_dim)

    #=============Decoder=============#
    self.decoder_input = nn.Linear(self.latent_dim, self.fc_input_size)
    self.unpool = nn.MaxUnpool2d(kernel_size = (self.k_pool_size,1))
    self.deconv1 = nn.ConvTranspose2d(self.channels, 1, (1, self.channels), stride=(1, 1))
    self.deconv2 = nn.ConvTranspose2d(16, 1, (self.timestamps//60,1), stride=(1,1))
    self.deconv3 = nn.ConvTranspose2d(16, 16, (self.timestamps//60,1), stride=(1,1))

    #===============GAN===============#
    fcs = ['fc_crit0','fc_crit1','fc_crit2','fc_crit3','fc_crit4']
    bns = ['bn_crit0','bn_crit1','bn_crit2','bn_crit3','bn_crit4']
    for i in range(self.gan_depth-1):
        self.add_module(fcs[i], nn.Linear(self.latent_dim//2**(i), self.latent_dim//2**(i+1)))
        self.add_module(bns[i], nn.BatchNorm1d(num_features=self.latent_dim//2**(i+1)))
    self.add_module(fcs[self.gan_depth-1], nn.Linear(self.latent_dim//2**(self.gan_depth-1), 1))
    
    #============Classifier============#
    self.fc_clf = nn.Linear(self.latent_dim, 1)
    self.fc_clf_discr1 = nn.Linear(self.latent_dim, 2)
    self.fc_clf_discr2 = nn.Linear(self.latent_dim, 3)
    self.fc_clf_discr3 = nn.Linear(self.latent_dim, 4)
```
Do not forget to modify the encode and decode functions the same way.

In forward, define the self.xxx for variables used in loss functions and modify the different pred_class depending on your task
```
def forward(self, inp: Tensor, **kwargs) -> Tensor:        
    if inp.dim() < 4:
        inp = inp.unsqueeze(1)
    self.ae_input = inp # needed to compute reconstruction loss

    x = inp.permute(0, 1, 3, 2)
    zi, ind1, ind2, in1, in2 = self.encode(x)

    self.zi = zi # needed to further display the latent space

    zi_gan = zi.view(-1, self.latent_dim, 1)
    self.gan_fake = self.latent_gan(zi_gan)
    z = torch.randn_like(zi_gan)
    self.gan_real = self.latent_gan(z)

    decoded = self.decode(zi, ind1, ind2, in1, in2)
    self.decoded = decoded.permute(0, 1, 3, 2)

    self.pred = self.fc_clf(zi).to(dev)

    self.pred_class = F.softmax(self.fc_clf_discr1(zi)).to(dev)
    self.pred_class2 = F.softmax(self.fc_clf_discr2(zi)).to(dev).argmax(dim=1)
    self.pred_class3 = F.softmax(self.fc_clf_discr3(zi)).to(dev).argmax(dim=1)
    
    preds = torch.cat([self.pred] * config['nb_of_labels'], dim=1) # force the same shape as the labels

    return  preds
```
Note: *preds* should have the same shape as the labels you gave as output of your *dls*

Modify the *classif_loss_function* to fit your labels
```
def classif_loss_func(self, output, target):
    self.targ1 = target[:,1].to(dev).type(torch.long)
    self.targ2 = target[:,2].to(dev).type(torch.long)
    self.targ3 = target[:,3].to(dev).type(torch.long)
    self.lab3 = target[:,4].to(dev).type(torch.float32)
    self.lab4 = target[:,5].to(dev).type(torch.float32)
    target = target[:,0].to(dev).type(torch.float32)

    delta = .5
    huber = nn.HuberLoss(delta=delta)
    bce = nn.CrossEntropyLoss()

    self.recons_loss = (huber(self.decoded, self.ae_input) +
        2*huber(self.decoded.std(dim=-1), self.ae_input.std(dim=-1)) # avoid the decoded signal to stay at 0
        ).to(device)

    # Curriculum learning on each severity metrics independently
    self.area_loss = bce(self.pred_class, self.targ1)# + .2*huber(self.pred_class, self.targ1)
    if self.level == 0:
        self.gather_loss = self.ordinal_loss(self.pred, target)# + .2*huber(output, target)
    if self.level == 0 or self.level >= 2:
        self.duration_loss = bce(self.pred_class, self.targ2)# + .2*huber(self.pred_class, self.targ2)
    if self.level == 0 or self.level == 3:
        self.arousal_loss = bce(self.pred_class, self.targ3)
            
    # Use of global loss
    if self.level == 0:
        loss = self.gather_loss
        if self.global_loss:
            loss = loss + \
                    .01*self.area_loss + .01*self.duration_loss + .005*self.arousal_loss +\
                    .005*self.recons_loss
            # print(f'losses: {loss, self.area_loss, self.duration_loss, self.arousal_loss, self.recons_loss}')
    elif self.level == 1:
        loss = self.area_loss
        if self.global_loss:
            loss = loss + .005*self.recons_loss
    elif self.level == 2:
        loss = self.duration_loss
        if self.global_loss:
            loss = loss + \
                    .1*self.area_loss + .005*self.recons_loss
    elif self.level == 3:
        loss = self.arousal_loss
        if self.global_loss:
            loss = loss + \
                    .1*self.duration_loss + .1*self.area_loss + .005*self.recons_loss

    # Final loss
    self.simple_loss = loss
    self.ord_loss = self.ordinal_loss(self.pred, target)
    loss = loss + .1*self.ord_loss

    return loss
```

### 5. Run the main script
Run the following command while being in the *xAAEnet* directory
```
python main.py
```
The results will be stored in the [/results](https://github.com/numediart/xAAEnet/results/) folder. The most important figure being *z_result_tsne*. It represents the latent space in 2D with the most discriminant direction represented by a red arrow, like in this example:
![TSNE representation](https://github.com/numediart/xAAEnet/blob/main/results/z_result_tsne.png)

## Data
Each trial in the dataset is composed of 23 channels and 3001 timestamps, as shown on Figure 1
![alt text](https://github.com/numediart/xAAEnet/blob/main/data/data.png)
Fig. 1. Data overview. Example of a preprocessed 60-seconds trial with OSA
event. Channels: 1) nasal airflow, 2-3) abdominal-thoracic respiratory motions,
4) oxygen saturation, 5-6) electrooculograms, 7) pulse rate variability, 8)
abdominal-thoracic motions phase shift, 9-23) EEG signal of the 3 selected
electrodes at different frequency ranges.

## Preprocessing
The EEG signals have been preprocessed
following the COBIDAS MEEG recommendations from the
Organization for Human Brain Mapping (OHBM) [1]. Trials
significantly affected by ocular artifacts have been excluded
from the database, based on the correlation between the EOG
and the FP1 signals. Trials with non-physiological amplitudes
are also excluded, based on their peak-to-peak voltage (VPP):
VP-P < 10<sup>-7</sup>V and VP-P > 6 ∗ 10<sup>-4</sup>V are excluded. A
baseline correction was applied using a segment of 10 seconds
preceding each trial as the baseline. The EEG delta band powe7
being the most varying frequency band during sleep apneahypopnea
occurrence [2], we focused our analysis on low
frequency EEG components by filtering the signals into 2Hz
narrow bands: 0-2Hz, 2-4Hz, 4-6Hz, 6-8Hz, and 8-10Hz. We
also rejected trials based on physiological fixed range criteria
on VP-P for EOG and SAO2 signals, moreover trials with
VAB, VTH and NAF2P statistical outliers in amplitude are
rejected. Two additional signals have been computed from the
aforementioned recorded signals: 1) the Pulse Rate Variability
(PRV) being the difference between a PR sample and the
next one, and 2) the belts phase shift (Pshift), computed as
the sample by sample phase difference between VAB and
VTH phase signals, as suggested by Varady et al. [3]. The
normalization has been performed by channel independently
as a z-score normalization with clamping in the [-3; 3] range.
After the exclusion and preprocessing phases, the final dataset
is composed of 6992 OSA trials from 60 patients divided
into a training set of 4660 trials from 48 patients, namely
the trainset, and a validation set of 2332 trials from the 12
remaining patients, namely the testset.

[1] C. Pernet, M. I. Garrido, A. Gramfort, N. Maurits, C. M. Michel,
E. Pang, R. Salmelin, J. M. Schoffelen, P. A. Valdes-Sosa, and A. Puce,
“Issues and recommendations from the OHBM COBIDAS MEEG committee
for reproducible EEG and MEG research,” Nature Neuroscience,
vol. 23, no. 12, pp. 1473–1483, Dec. 2020, number: 12 Publisher: Nature
Publishing Group.

[2] C. Shahnaz, A. T. Minhaz, and S. T. Ahamed, “Sub-frame based apnea
detection exploiting delta band power ratio extracted from EEG signals,”
in 2016 IEEE Region 10 Conference (TENCON), Nov. 2016, pp. 190–
193, iSSN: 2159-3450.

[3] P. Varady, S. Bongar, and Z. Benyo, “Detection of airway obstructions
and sleep apnea by analyzing the phase relation of respiration movement
signals,” IEEE Transactions on Instrumentation and Measurement,
vol. 52, no. 1, pp. 2–6, Feb. 2003, conference Name: IEEE Transactions
on Instrumentation and Measurement.

## Architecture
The xAAEnet architecture is a variation of the [xVAEnet](https://github.com/numediart/xVAEnet), with several key differences. While both models have an encoder-decoder structure with a latent space in between, the xAAEnet's latent block has a simpler design, consisting of only one dense layer and one batch normalization layer. This block does not use any reparameterization technique, in contrast to the xVAEnet, which employs a variational autoencoder approach. Additionally, the xAAEnet's final block is a regressor block that includes a single-layer perceptron with one output, and no activation function. These changes were made to adapt the model for the specific task of severity scoring in obstructive sleep apnea, and to simplify the training process.
