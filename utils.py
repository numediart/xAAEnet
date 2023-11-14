# Luca La Fisca
# ------------------------------
# Copyright UMONS (C) 2022

from fastai.tabular.all import *
from tsai.all import *
from torch import nn
from fastai.vision.gan import *
from torch.autograd import Variable
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import kendalltau


device = torch.device('cuda:1')

dev = torch.device('cpu')

class norm_batch(Transform):
    def __init__(self, eps=1e-08, n_extra=0) :
        self.eps = eps
        self.n_extra = n_extra
    def encodes(self, t:torch.Tensor):
        try:
            if self.n_extra > 0:
                extra = t[:,:,-self.n_extra:]
                t = t[:,:,:-self.n_extra]
            mean = torch.nanmean(t, dim=2)
            std = torch.clamp_min(torch.std(t, dim=2), self.eps)
            out = torch.stack([torch.vstack([(t[j,i,:]-mean[j,i])/torch.clamp_min(std[j,i], self.eps)
                                 for i in range(t.shape[1])]) 
                                 for j in range(t.shape[0])],dim=0)
            out = torch.clamp_min(out,-3)
            out = torch.clamp_max(out,3)
            if self.n_extra > 0:
                out = torch.dstack((out,extra))
        except:
            out = t
        return out.to(device)

class LossAttrMetric(Metric):
    def __init__(self, attr):
        self.attr_name = attr
        self.vals = []
    def reset(self):
        self.vals = []
    def accumulate(self, learn):
        setattr(self, self.attr_name, getattr(learn, self.attr_name))
        self.vals.append(getattr(self, self.attr_name))
    @property
    def value(self):
        return torch.mean(torch.tensor(self.vals))
    @property
    def name(self):
        return self.attr_name

class CheckNorm(Callback):
    order=1
    def __init__(self, norm_y=False):
        self.norm_y = norm_y
    def before_batch(self):
        clamp_val = 3
        if torch.any(self.xb[0].max() > clamp_val) or torch.any(self.xb[0].min() < -clamp_val):
            self.xb[0][:] = torch.clamp_min(self.xb[0][:], -clamp_val)
            self.xb[0][:] = torch.clamp_max(self.xb[0][:], clamp_val)

        if self.norm_y and (torch.any(self.yb[0].max() > clamp_val) or torch.any(self.yb[0].min() < -clamp_val)):
            self.yb[0][:] = torch.clamp_min(self.yb[0][:], -clamp_val)
            self.yb[0][:] = torch.clamp_max(self.yb[0][:], clamp_val)


class FreezeEncoderCallback(Callback):        
    def before_batch(self):
        freeze_layers = 11 #the encoder part corresponds to the 11 first layers of the model
        for child in list(self.children())[:freeze_layers]:
            for p in child.parameters():
                if self.gen_train:
                    p.requires_grad_(True)
                else:
                    p.requires_grad_(False)

class UnfreezeFcCrit(Callback):
    def __init__(self, switch_every: int=7):
        self.switch_every = switch_every  
    # def before_batch(self):
    #     if self.training and (self.iter + 1) % (self.acc_factor * 3) == 0:
    def before_train(self):
        # print('Im in unfreeze')
        # print(self.training, self.epoch)
        if (self.epoch + 1) % self.switch_every == 0:
            self.learn.model.train_crit = True
            for name, param in self.learn.model.named_parameters():
                if "fc_crit" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
        # elif self.training and (self.iter + 1) % (self.acc_factor * 4) == 0:
        elif (self.epoch + 1) % self.switch_every == 1:
            self.learn.model.train_crit = False
            for name, param in self.learn.model.named_parameters():
                if "fc_crit" in name:
                    param.requires_grad_(False)
                else:
                    param.requires_grad_(True)
    # def after_loss(self):
    #     # print(f'im in after_loss of unfreeze, train_crit is {self.train_crit}\n \
    #     #     the self.training is {self.training}')
    #     if self.train_crit:
    #         adversarial_loss = nn.BCELoss()
    #         valid = Variable(Tensor(self.zi.shape[0], 1).fill_(1.0), requires_grad=False).to(device)
    #         fake = Variable(Tensor(self.zi.shape[0], 1).fill_(0.0), requires_grad=False).to(device)
    #         # Measure discriminator's ability to classify real from generated samples
    #         self.real_loss = adversarial_loss(self.gan_real, valid)
    #         self.fake_loss = adversarial_loss(self.gan_fake, fake)
    #         self.learn.model.adv_loss_cbs = 0.6 * self.real_loss + 0.4 * self.fake_loss
    #         self.learn.loss_grad = self.learn.model.adv_loss_cbs
    #         # print(f'iteration number {self.iter}, final loss is {self.loss}\n \
    #         #     real loss = {self.real_loss}, fake loss = {self.fake_loss}')
        # print(f'in unfreeze crit, train_crit = {self.learn.model.train_crit}')


class SwitchAttribute(Callback):
    def __init__(self, attribute_name: str, switch_every: int, original_state: bool=False):
        self.attribute_name = attribute_name
        self.switch_every = switch_every
        self.original_state = original_state

    # def before_train(self):
    #     self.original_state = getattr(self.learn.model, self.attribute_name)
        # print(f'the attribute {self.attribute_name} is set to {self.original_state}\n \
        #     epoch numbe {self.epoch}, switch every {self.switch_every}')
    def before_train(self):
        self.original_state = getattr(self.learn.model, self.attribute_name)
        if (self.epoch+1) % self.switch_every == 0 or (self.epoch+1) % self.switch_every == 1:
            # self.learn.model.global_loss = not self.original_state
            # self.learn.model.global_loss = True
            setattr(self.learn.model, self.attribute_name, not self.original_state)
        # if (self.epoch+1) % self.switch_every == 1:
        #     # self.learn.model.global_loss = not self.original_state
        #     # self.learn.model.global_loss = False
        #     setattr(self.learn.model, self.attribute_name, not self.original_state)
        # print(f'in switch attribute, gen_train = {self.original_state}')

#Define a callback to get the latent space
class GetLatentMVAR(Callback):
    def __init__(self, cycle_len=None):
        self.cycle_len_init = cycle_len
    def before_validate(self):
        self.cycle_len = ifnone(self.cycle_len_init,self.n_epoch)
    def after_batch(self):
        if not self.training:
            if (self.epoch+1) % self.cycle_len == 0:
                if not hasattr(self, 'zi_valid') or self.zi_valid.numel() == 0:
                    if hasattr(self, 'zi'):
                        self.learn.zi_valid = self.zi
                    else:
                        self.learn.zi_valid = self.generator.zi
                else:
                    if hasattr(self, 'zi'):
                        self.learn.zi_valid = torch.vstack((self.learn.zi_valid,self.zi))
                    else:
                        self.learn.zi_valid = torch.vstack((self.learn.zi_valid,self.generator.zi))


class GetLatentSpace(Callback):
    def __init__(self, cycle_len=None):
        self.cycle_len_init = cycle_len
                
    def before_validate(self):
        self.cycle_len = ifnone(self.cycle_len_init,self.n_epoch)
        if (self.epoch+1)% self.cycle_len == 0 and (hasattr(self,'fc_mu') \
            or hasattr(self,'generator')):
            print("GetLatentSpace, before_validate, Hook starts, Epoch: "+str(self.epoch))
            if (self.epoch+1) % self.cycle_len == 0:
                def getActivation(name):
                    # the hook signature
                    def hook(model, input, output):
                        self.activation[name] = output.detach()
                    return hook
                self.activation = {}
                try:
                    self.mu = self.learn.model.fc_mu.register_forward_hook(getActivation('mu'))
                    self.std = self.learn.model.fc_var.register_forward_hook(getActivation('std'))

                except:  #for GAN
                    self.mu = self.learn.generator.fc_mu.register_forward_hook(getActivation('mu'))
                    self.std = self.learn.generator.fc_var.register_forward_hook(getActivation('std'))
                    
    def after_batch(self):
        if not self.training:
            if (self.epoch+1) % self.cycle_len == 0 and (hasattr(self,'fc_mu') or hasattr(self,'generator')):
                if hasattr(self, 'mu_valid'):
                    self.learn.mu_valid = torch.vstack((self.learn.mu_valid,self.activation['mu']))
                    self.learn.std_valid = torch.vstack((self.learn.std_valid,self.activation['std']))
                    if hasattr(self, 'zi'):
                        self.learn.zi_valid = torch.vstack((self.learn.zi_valid,self.zi))
                    else:
                        self.learn.zi_valid = torch.vstack((self.learn.zi_valid,self.generator.zi))

                else:
                    self.learn.mu_valid = self.activation['mu']
                    self.learn.std_valid = self.activation['std']
                    if hasattr(self, 'zi'):
                        self.learn.zi_valid = self.zi
                    else:
                        self.learn.zi_valid = self.generator.zi

            elif (self.epoch+1) % self.cycle_len == 0:
                if not hasattr(self, 'zi_valid') or self.zi_valid.numel() == 0:
                    if hasattr(self, 'zi'):
                        self.learn.zi_valid = self.zi
                    else:
                        self.learn.zi_valid = self.generator.zi
                else:
                    if hasattr(self, 'zi'):
                        self.learn.zi_valid = torch.vstack((self.learn.zi_valid,self.zi))
                    else:
                        self.learn.zi_valid = torch.vstack((self.learn.zi_valid,self.generator.zi))
                    
            
    def after_validate(self):
        if (self.epoch+1)% self.cycle_len == 0 and (hasattr(self,'fc_mu') or hasattr(self,'generator')):
            std = torch.exp(0.5 * self.learn.std_valid)
            eps = torch.randn_like(std)
            self.learn.zs = eps * std + self.learn.mu_valid
            print(f'shape of zs is {self.zs.shape}')
            del self.learn.std_valid, self.learn.mu_valid
            torch.cuda.empty_cache()
            self.mu.remove()
            self.std.remove()
            

#Define a callback to modify the target zs
class ChangeTargetData(Callback):
    def __init__(self,cycle_len=None,splitter=None,getters=None): 
        self.cycle_len = cycle_len
        self.splitter = splitter
        self.getters = getters

    def after_epoch(self):
        if (self.epoch+1) % self.cycle_len == 0 and self.train_iter>0:
            print("In epoch "+str(self.epoch)+", let's change the target latent space")
            x = torch.stack((list(zip(*self.dls.valid_ds))[0]),dim=0)
            y = self.zs

            torch.cuda.set_device(device)
            tmp_cbs = self.cbs[-2:]
            self.learn.cbs = self.cbs[:-2]
            self.learn.add_cb(GetLatentSpace(cycle_len=1))
            self.learn.get_preds(ds_idx=0,inner=True)
            self.learn.cbs = self.cbs[:-1]
            self.learn.add_cbs(tmp_cbs)
            y = torch.vstack((self.zs,y))
            x = torch.vstack((torch.stack((list(zip(*self.dls.train_ds))[0]),dim=0),x))
            print(f'shapes of x/y in change target: {x.shape}, {y.shape}')
            print(f'before changing yb, shape: {self.yb[0].shape}')
            dblock = DataBlock(blocks=(TSTensorBlock,TSTensorBlock),
                              splitter=self.splitter,
                              getters=self.getters,
                              batch_tfms=norm_batch(n_extra=1))
            src = itemify(x.to(dev),y.to(dev))
            self.learn.dls = dblock.dataloaders(src,bs=16,val_bs=32)
            print(f'after changing y, shape: {self.yb[0].shape}')
            print("before changing zs, the loss is: "+str(self.loss))


class TrainClassif(Callback):
    order = 1
    def __init__(self, gan_depth: int=3):
        self.gan_depth = gan_depth
    def before_train(self):
        cycle_len = 5
        # fcs = ['fc_crit0','fc_crit1','fc_crit2','fc_crit3','fc_crit4']

        if (self.epoch+1) % cycle_len < 2:
            # uncomment to unfreeze frozen classifier layers 
            # self.learn.model.fc_clf = self.learn.model.fc_clf.requires_grad_(True)
            # self.learn.model.fc_clf2 = self.learn.model.fc_clf2.requires_grad_(True)
            # self.learn.model.fc_clf3 = self.learn.model.fc_clf3.requires_grad_(True)
            for i in range(self.gan_depth-1):
                # getattr(self.learn.model,f'fc_crit{i}') = getattr(self.learn.model,f'fc_crit{i}').requires_grad_(True)
                setattr(self.learn.model,f'fc_crit{i}.requires_grad_',True)

            # self.learn.model.fc_crit = self.learn.model.fc_crit0.requires_grad_(True)
            # self.learn.model.fc_crit2 = self.learn.model.fc_crit2.requires_grad_(True)
            # self.learn.model.fc_crit3 = self.learn.model.fc_crit3.requires_grad_(True)
            print("!!!!!!!!!!! GRAD SET TO TRUE !!!!!!!!!!!!!")
        elif (self.epoch+1) % cycle_len == 2:
            # uncomment to freeze classifier layers
            # self.learn.model.fc_clf = self.learn.model.fc_clf.requires_grad_(False)
            # self.learn.model.fc_clf2 = self.learn.model.fc_clf2.requires_grad_(False)
            # self.learn.model.fc_clf3 = self.learn.model.fc_clf3.requires_grad_(False)
            for i in range(self.gan_depth-1):
                # getattr(self.learn.model,f'fc_crit{i}') = getattr(self.learn.model,f'fc_crit{i}').requires_grad_(False)
                setattr(self.learn.model,f'fc_crit{i}.requires_grad_',True)

            # self.learn.model.fc_crit = self.learn.model.fc_crit.requires_grad_(False)
            # self.learn.model.fc_crit2 = self.learn.model.fc_crit2.requires_grad_(False)
            # self.learn.model.fc_crit3 = self.learn.model.fc_crit3.requires_grad_(False)
            print("!!!!!!!!!!! GRAD SET TO FALSE !!!!!!!!!!!!!")

    def after_loss(self):
        cycle_len = 5
        ce = nn.CrossEntropyLoss()
        if (self.epoch+1) % cycle_len < 2:
            #compute loss with equal weight for each single loss
            VAE_weight = .5
            classif_weight = 1/3
            self.learn.tk_mean = 1 - self.tk_mean
            self.learn.loss = ((1-classif_weight)*(VAE_weight*
                    ((1-self.kld_weight)*self.recons_loss + #Reconstruction loss
                    self.kld_weight*self.kld_loss) + #Kullback-Leibler loss
                    (1-VAE_weight)*self.tk_mean) + #GAN generator loss
                    # classif_weight*self.classif_loss) #Classif loss
                    self.classif_loss) #Classif loss
        # elif self.nclass == 2 and (self.epoch+1) % 2 == 1:
        #     self.learn.loss = 1/2 * ce(self.pred_class,self.targ1) + 1/2 * ce(self.pred_class,self.targ2)
        # elif self.nclass == 3 and (self.epoch+1) % 3 == 1:
        #     self.learn.loss = (1/3 * ce(self.pred_class,self.targ1) + 1/3 * ce(self.pred_class,self.targ2) +
        #                         1/3 * ce(self.pred_class,self.targ3))

    def after_epoch(self):
        print("final loss is: ", str(self.loss))
        print("single losses are: reconstruction: ", str(self.recons_loss), " KLD: ", str(self.kld_loss),
            " GAN: ", str(self.tk_mean), " classif: ", str(self.classif_loss))


from sklearn.linear_model import LinearRegression
from collections import Counter
from scipy.ndimage import convolve1d, gaussian_filter1d

# Taken from https://github.com/YyzHarry/imbalanced-regression
def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window


# Get the normalized scores for area and duration
def get_normalized_scores(display=False):
    # for desaturation area
    area_db = []
    with open("/home/JennebauffeC/pytorchVAE/label_area.txt") as f :
        ligne = f.readline().rstrip(" \n")
        while ligne:
            ligne = list(map(int, ligne.split(" ")))
            patient = ligne[0]
            area = ligne[1:]
            area_db += area
            ligne = f.readline().rstrip(" \n")
    print(len(area_db))

    med_area = np.median(area_db)
    norm_area = np.clip((area_db - med_area) / med_area, -1, 1)

    print(med_area)
    print(norm_area.min())
    print(norm_area.max())

    np.save('/home/JennebauffeC/pytorchVAE/norm_area_db.npy', norm_area)

    # for apnea duration
    duration_db = []
    with open("/home/JennebauffeC/pytorchVAE/label_duree.txt") as f :
        ligne = f.readline().rstrip(" \n")
        while ligne:
            ligne = list(map(float, ligne.split(" ")))
            patient = ligne[0]
            duration = ligne[1:]
            duration_db += duration
            ligne = f.readline().rstrip(" \n")
    print(len(duration_db))

    med_duration = np.median(duration_db)
    norm_duration = np.clip((duration_db - med_duration) / med_duration, -1, 1)

    print(med_duration)
    print(norm_duration.min())
    print(norm_duration.max())

    np.save('/home/JennebauffeC/pytorchVAE/norm_duration_db.npy', norm_duration)

    if display:
        bins = np.arange(-1,1,.01) # fixed bin size
        plt.hist(norm_duration, bins=bins)
        plt.axvline(x = int(np.median(norm_duration)), color = 'r', label = f'median={str(int(np.median(norm_duration)))}')
        plt.legend()
        plt.show()


# Compute the regularized linear regression of the latent space wrt the labels
def distrib_regul_regression(z, target, nbins: int=100, get_reg: bool=False):
    bin_edges = np.linspace(target.min(), target.max(), nbins+1)
    # Assign each value in the data to its corresponding category based on the bin edges
    labels = np.digitize(target, bin_edges)
    bin_index_per_label = [int(label) for label in labels]

    # calculate empirical (original) label distribution: [Nb,]
    # "Nb" is the number of bins
    Nb = max(bin_index_per_label) + 1
    num_samples_of_bins = dict(Counter(bin_index_per_label))
    emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(Nb)]

    # lds_kernel_window: [ks,], here for example, we use gaussian, ks=5, sigma=2
    lds_kernel_window = get_lds_kernel_window(kernel='gaussian', ks=5, sigma=2)
    # calculate effective label distribution: [Nb,]
    eff_label_dist = convolve1d(np.array(emp_label_dist), weights=lds_kernel_window, mode='constant')

    # Use re-weighting based on effective label distribution, sample-wise weights: [Ns,]
    eff_num_per_label = [eff_label_dist[bin_idx] for bin_idx in bin_index_per_label]
    weights = [np.float32(1 / x) for x in eff_num_per_label]

    reg = LinearRegression().fit(z, target, sample_weight=weights)
    out = np.dot(z, reg.coef_) + reg.intercept_

    if get_reg:
        return out, reg
    else:
        return out

# Create an histogram-like plot of the mean target value by x-bins
def hist_lab(preds, target, nbins=42, reg=True, show: bool=True, get_met: bool=False):
    data = np.vstack((preds, target)).T

    # séparer les colonnes de vos données
    col1 = np.array([row[0] for row in data])
    col2 = np.array([row[1] for row in data])

    # calculer les bins
    if reg:
        bins = np.percentile(col1, np.linspace(0, 100, nbins))
    else:
        bins = np.linspace(min(col1), max(col1), nbins)

    # calculer les moyennes pour chaque bin
    bin_means = [np.mean(col2[(col1 >= bins[i]) & (col1 < bins[i+1])]) for i in range(len(bins)-1)]
    # normalize in [-1,1]
    bin_means = (bin_means - min(bin_means))/(max(bin_means)-min(bin_means)) * 2 - 1

    ord_met = ordinal_metric(bin_means)

    if show:
        # tracer le barplot (avec bins regularisées)
        bins = np.linspace(min(col1), max(col1), nbins)
        plt.bar(bins[:-1], bin_means, width=bins[1]-bins[0], color='#1f77b4')
        # sns.barplot(x=bins[:-1], y=bin_means, color='#0077bb') #barwidth=bins[1]-bins[0],

    if get_met:
        return ord_met
    else:
        print(f'ordinal metric = {ord_met}')


def ordinal_metric(bin_means):
    ord_met = 0.0
    rnd_met = 0.0
    if np.sum(np.diff(bin_means)) < 0:
#         print("bin_means reverted")
        bin_means = bin_means[::-1]
    # create a random ordered bins vector for comparison
    rnd_bins = bin_means.copy()
    np.random.shuffle(rnd_bins)
    if np.sum(np.diff(rnd_bins)) < 0:
        rnd_bins = rnd_bins[::-1]
    # compute the metric
    for i in range(len(bin_means)-1):
        diff = bin_means[i+1:] - bin_means[i]
        diff[diff > 0] = 0.0
        ord_met += np.mean(-diff)
        #for the random bins
        diff = rnd_bins[i+1:] - rnd_bins[i]
        diff[diff > 0] = 0.0
        rnd_met += np.mean(-diff)
    return ord_met / rnd_met

def get_kendall_tau(z,lab_gather,filename,safe_ratio=0.0):
    y_pred = distrib_regul_regression(z.cpu().detach().numpy(), lab_gather)
    safepred = safe_ratio*lab_gather + (1-safe_ratio)*y_pred
    y_sort, idx_sort = torch.tensor(y_pred).sort()
    lab_sort, _ = lab_gather.sort()
    safelab = (1-safe_ratio)*lab_gather[idx_sort] + safe_ratio*lab_sort
    _, idx_back = idx_sort.sort()
    safelab = safelab[idx_back]
    # print(f'ordinal loss: {learn.model.ordinal_loss(torch.tensor(y_pred).to(dev),lab_gather.to(dev),mydev=dev)}')

    # Compute Kendall Tau distance
    print(f'y pred min/max: {safepred.min(), safepred.max()}')
    print(f'lab_gather min/max: {lab_gather.min(), lab_gather.max()}')
    tau, _ = kendalltau(safepred, lab_gather)
    print(f'Kendall rank correlation coefficient = {tau}')


def plot_results(z,lab_gather,learn,filename,nbins=24,safe_ratio=0.0):
    # pal = sns.color_palette('YlOrBr_r',n_colors=8)
    # mypal = np.tile(pal[0],(len(lab_all),1))
    # for i,l in enumerate(lab_all):
    #     mycol = int(l.item())
    #     mypal[i] = pal[mycol]

    sns.set(rc={'figure.figsize':(11.7,8.27)})
    y_pred = distrib_regul_regression(z.cpu().detach().numpy(), lab_gather)
    y_sort, idx_sort = torch.tensor(y_pred).sort()
    lab_sort, _ = lab_gather.sort()
    safelab = (1-safe_ratio)*lab_gather[idx_sort] + safe_ratio*lab_sort
    _, idx_back = idx_sort.sort()
    safelab = safelab[idx_back]
    print(f'ordinal loss: {learn.model.ordinal_loss(torch.tensor(y_pred).to(dev),lab_gather.to(dev),mydev=dev)}')

    # Compute the mean error and compare to the mean error of randomly sorted trials
    y_sort, idx_sort = torch.Tensor(y_pred).sort()
    idx_rnd = np.random.permutation(np.arange(0,len(idx_sort)))
    lab_sort = lab_gather[idx_sort]
    lab_rnd = lab_gather[idx_rnd]
    error_sum = 0
    rnd_sum = 0
    for i in range(1,len(y_pred)):
        error_sum += -torch.sum(torch.min(lab_sort[i] - lab_sort[:i], torch.tensor(0.0))) / \
                      torch.sum(torch.abs(lab_sort[i] - lab_sort[:i]))
        rnd_sum += -torch.sum(torch.min(lab_rnd[i] - lab_rnd[:i], torch.tensor(0.0))) / \
                    torch.sum(torch.abs(lab_rnd[i] - lab_rnd[:i]))
    error_mean = error_sum/len(y_pred)
    rnd_mean = rnd_sum/len(y_pred)
    print(f'Mean error = {error_mean}, Random mean error = {rnd_mean}')

    # Plot useful figures
    diverging_norm = mcolors.TwoSlopeNorm(vmin=safelab.min(),vcenter=0.0,vmax=safelab.max())
    mapper = plt.cm.ScalarMappable(norm=diverging_norm, cmap='YlOrBr_r')
    colors = mapper.to_rgba(safelab.numpy())

    plt.figure()
    nbins = nbins
    # ord_met = hist_lab(y_pred, lab_gather, nbins, get_met=True)
    ord_met = hist_lab(y_pred, safelab, nbins, get_met=True)
    print(f'ord_met = {ord_met}')
    np.save(f'results/ord_met_{filename}.npy', ord_met)
    # plt.title("Mean label values along the severity direction")
    plt.xlabel("Latent Severity Scale")
    plt.ylabel("Mean Hand-made Score S_h")
    plt.xticks(ticks=[y_pred.min(),y_pred.max()], labels=[0,1])
    plt.savefig("results/z_"+str(filename)+"_regression_hist")

    plt.figure()
    sns.scatterplot(x=y_pred, y=np.random.uniform(-50000, 50000,len(y_pred)), c=colors)
    plt.title("1-D distribution of the sorted predictions")
    plt.xticks(ticks=[y_pred.min(),y_pred.max()], labels=[0,1])
    plt.savefig("results/z_"+str(filename)+"_1D")

    from sklearn.manifold import TSNE
    tsne = TSNE(random_state=42)
    predictions_embedded = tsne.fit_transform(z.cpu().detach().numpy())

    #Compute linear regression from 2D space
    y_pred_embed = distrib_regul_regression(predictions_embedded, lab_gather)

    y_sort, idx_sort = torch.tensor(y_pred_embed).sort()
    safelab = (1-safe_ratio)*lab_gather[idx_sort] + safe_ratio*lab_sort
    _, idx_back = idx_sort.sort()
    safelab = safelab[idx_back]

    # Calculate the mean of x and y for the darkest and lightest colors
    q1, q3 = np.percentile(safelab, [25, 75])
    dark_mask = safelab <= q1
    light_mask = safelab >= q3
    dark_mean = np.mean(predictions_embedded[dark_mask, :], axis=0)
    light_mean = np.mean(predictions_embedded[light_mask, :], axis=0)
    # Get the difference between dark_mean and light_mean
    diff = light_mean - dark_mean
    # Calculate the slope
    m = diff[1] / diff[0]
    m = -1/m
    # Calculate the intercept
    b = dark_mean[1] - m * dark_mean[0]
    b = 0

    # Calculer les points de début et de fin de la droite régressée
    x, y = predictions_embedded[:, 0], predictions_embedded[:, 1]
    max_x = np.max(np.abs(x)) - 5
    max_y = np.max(np.abs(y)) - 5
    if max_x >= max_y:
        x_min, x_max = -max_x, max_x
    else:
        x_min, x_max = -np.abs((-max_y - b) / m), np.abs((max_y - b) / m)
    y_min, y_max = x_min * m + b, x_max * m + b
    # Define start/end point of the arrow
    start = (x_min,y_min)
    end = (x_max,y_max)

    # Sort the trials along the severity direction 
    x_proj = []
    for x, y in predictions_embedded:
        x_proj.append((x + m * y - m * b) / (1 + m ** 2))
    x_proj = np.array(x_proj)

    if dark_mean[0] < light_mean[0]:
        _, idx_sort = torch.tensor(x_proj).sort()
    elif dark_mean[0] > light_mean[0]:
        _, idx_sort = torch.tensor(-x_proj).sort()
    else:
        raise ValueError("Severity direction is vertical")

    lab_sort, _ = lab_gather.sort()
    # safelab = (1-safe_ratio)*lab_gather[idx_sort] + safe_ratio*lab_sort
    # _, idx_back = idx_sort.sort()
    # safelab = safelab[idx_back]
    diverging_norm = mcolors.TwoSlopeNorm(vmin=safelab.min(),vcenter=0.0,vmax=safelab.max())
    mapper = plt.cm.ScalarMappable(norm=diverging_norm, cmap='YlOrBr_r')
    colors = mapper.to_rgba(safelab.numpy())

    fig, ax = plt.subplots()
    # sns.scatterplot(x=predictions_embedded[:,0], y=predictions_embedded[:,1], c=mypal)
    sns.scatterplot(x=predictions_embedded[:,0], y=predictions_embedded[:,1], c=colors)
    # Plot the line along the first principal component
    ax.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1], linewidth=3,
              head_width=10, head_length=10, fc='r', ec='r', length_includes_head=True)
    # Define x,y limits
    maxabs = np.max(np.abs(predictions_embedded)) + 5
    maxabs = np.max(np.abs([start[0],start[1],end[0],end[1]]))
    plt.xlim([-maxabs, maxabs])
    plt.ylim([-maxabs, maxabs])

    plt.title("Z representations in 2D using TSNE")
    plt.savefig("results/z_"+str(filename)+"_tsne")

    return x_proj