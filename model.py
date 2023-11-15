# Luca La Fisca
# ------------------------------
# Copyright UMONS (C) 2023

from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from fastai.data.all import *
import time
import numpy as np
import sys

# Load the config file
config_file = 'config.json'
with open(config_file, 'r') as file:
    config = json.load(file)

device = torch.device(config['device'])
dev = device

class stagerNetAAE(nn.Module):
    def __init__(self, typ: int=0, channels: int=23, timestamps: int=3001, nclass: int=1,
                    use_adv: bool=False, use_norm: bool=False, use_embed: bool=False, 
                    acc_factor: int=8, dropout_rate: float=0.5,
                    latent_dim: int=128, gan_depth: int=3, k_pool_size: int=13,
                    info_weight: float=.2, vae_weight: float=.1, classif_weight: float = .95,
                    epsilon:float=0.1, max_iter:int=100, level: int=0):
        super(stagerNetAAE, self).__init__()
        
        self.typ = typ
        self.nclass = nclass
        self.channels = channels #number of input channels (spatial)
        self.timestamps = timestamps #number of input timestamps (temporal)
        self.use_adv = use_adv
        self.use_norm = use_norm
        self.use_embed = use_embed
        self.acc_factor = acc_factor
        self.latent_dim = latent_dim #embed_dim
        self.k_pool_size = k_pool_size #embed_dim
        self.dropout_rate = dropout_rate
        self.info_weight = info_weight
        self.vae_weight = vae_weight
        self.classif_weight = classif_weight
        self.gan_depth = gan_depth
        self.epsilon, self.max_iter = epsilon, max_iter
        self.gen_train = True
        self.count_acc = 0
        self.level = level
        self.global_loss = False
        self.train_crit = False
        
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
#         if self.typ == 3 :
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

            
    def _get_fc_input_size(self, timestamps):
        x = torch.randn(16, 1, timestamps, self.channels)
        x,_,_,_,_ = self._forward_conv(x)
        # return x.view(x.size(0), -1).size(1)
        return x.shape[1]

    def _forward_conv(self, inp):
        inp = self.conv1(inp)
        inp = inp.permute(0, 3, 2, 1)
        inp = self.conv2(inp)
        input_mp1 = inp.detach().clone()
        inp, ind_maxpool1 = self.pool(inp)
        inp = F.relu(inp)
        inp = self.batchnorm1(inp)
        inp = self.conv3(inp)
        input_mp2 = inp.detach().clone()
        inp, ind_maxpool2 = self.pool(inp)
        inp = F.relu(inp)
        result = self.batchnorm2(inp)
        result = torch.flatten(result, start_dim=1)
        result = F.dropout(result, p=self.dropout_rate)
        return result, ind_maxpool1, ind_maxpool2, input_mp1, input_mp2

    def encode(self, inp: Tensor) -> List[Tensor]:
        result, ind_maxpool1, ind_maxpool2, input_mp1, input_mp2 = self._forward_conv(inp)

        z = self.fc_z(result)
        zi = F.relu(self.bn_lin(z))

        return [zi, ind_maxpool1, ind_maxpool2, input_mp1, input_mp2]
    
    def decode(self, z: Tensor, ind1, ind2, in1, in2) -> Tensor:
        x = self.decoder_input(z)        
        x = x.view(-1, 16, 13, self.channels) 
        x = self.unpool(x, indices=ind2, output_size = in2.size())
        x = F.relu(x)
        x = self.batchnorm2(x)
        x = self.deconv3(x)
        x = self.unpool(x, indices=ind1, output_size = in1.size())
        x = F.relu(x)
        x = self.batchnorm1(x)
        x = self.deconv2(x)
        x = x.permute(0, 3, 2, 1)
        result = self.deconv1(x)        
        return result

    def latent_gan(self, zi: Tensor) -> Tensor:
        x = zi.view(-1,self.latent_dim)
        for i in range(self.gan_depth-1):
            x = getattr(self,f'fc_crit{i}')(x)
            x = F.leaky_relu(getattr(self,f'bn_crit{i}')(x),negative_slope=0.2)
        x = getattr(self,f'fc_crit{self.gan_depth-1}')(x)
        x = F.sigmoid(x)
        return x

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
        # if self.level == 0 or self.level >= 2:
        self.pred_class2 = F.softmax(self.fc_clf_discr2(zi)).to(dev).argmax(dim=1)
        # if self.level == 0 or self.level == 3:
        self.pred_class3 = F.softmax(self.fc_clf_discr3(zi)).to(dev).argmax(dim=1)
        
        preds = torch.cat([self.pred] * config['nb_of_labels'], dim=1) # force the same shape as the labels

        return  preds
  

    def ae_loss_func(self, output, target):
        delta = .5
        huber = nn.HuberLoss(delta=delta)
        recons_loss = (huber(self.decoded, self.ae_input) +
                2*huber(self.decoded.std(dim=-1), self.ae_input.std(dim=-1)) # avoid the decoded signal to stay at 0
                )
        return recons_loss

    def ordinal_loss(self, preds, target, mydev=dev):
        # print(f'in ordinal: {preds, target}')
        # Comute the term by term differences
        order_diff = target.unsqueeze(1) - target.unsqueeze(0)
        pred_order_diff = preds.unsqueeze(1) - preds.unsqueeze(0)

        # Create a mask to exclude self-differences (diagonal elements)
        mask = torch.eye(preds.size(0)).bool()
        order_diff = order_diff[~mask].float()
        pred_order_diff = pred_order_diff[~mask].float()

        # Normalize the order_diff to avoid biased comparison
        if any(order_diff!=0) and any(pred_order_diff!=0):
            order_diff = (order_diff - order_diff.min()) / (order_diff.max() - order_diff.min())
            pred_order_diff = (pred_order_diff - pred_order_diff.min()) / (pred_order_diff.max() - pred_order_diff.min())

        loss = torch.abs(order_diff - pred_order_diff).mean()

        if loss.isnan():
            print(f'nan here: {preds, target,order_diff,pred_order_diff}')

        return loss
    
    def classif_loss_func(self, output, target):
        # print(f'The target type is {type(target)} with length: {len(target)}')
        # print('with shapes:')
        # for i in range(len(target[0])):
        #     print(f'target[{i}]: {target[:,i].shape}')
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

    def aae_loss_func(self, output, target):
        delta = .5
        huber = nn.HuberLoss(delta=delta)
        adversarial_loss = nn.BCELoss()
        if self.gen_train: #generator loss
            # Measures generator's ability to fool the discriminator
            valid = torch.ones_like(self.gan_fake, requires_grad=False).detach()
            self.adv_loss = adversarial_loss(self.gan_fake, valid)
            self.recons_loss = (huber(self.decoded, self.ae_input) +
                2*huber(self.decoded.std(dim=-1), self.ae_input.std(dim=-1)) # avoid the decoded signal to stay at 0
                )
            self.classif_loss = self.classif_loss_func(output, target)
            loss = 0.05 * self.adv_loss + 0.45 * self.recons_loss + 0.5 * self.classif_loss

        else: #discriminator loss
            # Measure discriminator's ability to classify real from generated samples
            valid = torch.ones_like(self.gan_real, requires_grad=False).detach()
            fake = torch.zeros_like(self.gan_fake, requires_grad=False).detach()
            self.real_loss = adversarial_loss(self.gan_real, valid)
            self.fake_loss = adversarial_loss(self.gan_fake, fake)
            loss = 0.6 * self.real_loss + 0.4 * self.fake_loss

        return loss


class stagerNetCritic(nn.Module):
    def __init__(self, latent_dim: int=128, gan_depth: int=3):
        super(stagerNetCritic, self).__init__()
        
        self.latent_dim = latent_dim #embed_dim
        self.gan_depth = gan_depth #depth of the dicriminator
        
        fcs = ['fc_crit0','fc_crit1','fc_crit2','fc_crit3','fc_crit4']
        bns = ['bn_crit0','bn_crit1','bn_crit2','bn_crit3','bn_crit4']
        for i in range(self.gan_depth-1):
            self.add_module(fcs[i], nn.Linear(self.latent_dim//2**(i), self.latent_dim//2**(i+1)))
            self.add_module(bns[i], nn.BatchNorm1d(num_features=self.latent_dim//2**(i+1)))

        self.add_module(fcs[self.gan_depth-1], nn.Linear(self.latent_dim//2**(self.gan_depth-1), 1))
        
    def discrim(self, zi: Tensor) -> Tensor:
        x = zi.view(-1,self.latent_dim)
        for i in range(self.gan_depth-1):
            x = getattr(self,f'fc_crit{i}')(x)
            x = F.leaky_relu(getattr(self,f'bn_crit{i}')(x),negative_slope=0.2)
        x = getattr(self,f'fc_crit{self.gan_depth-1}')(x)
        x = F.sigmoid(x)
        return x

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        if next(self.fc_crit0.parameters()).is_cuda:
            self.gan_class = self.discrim(input)
        else:
            self.gan_class = self.discrim(input.type(torch.FloatTensor).detach().cpu())
        return self.gan_class
