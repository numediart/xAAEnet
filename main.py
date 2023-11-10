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
from models.autoencoders import stagerNetAAE, stagerNetCritic
from utils import LossAttrMetric, FreezeEncoderCallback, ChangeTargetData, \
                GetLatentSpace, TrainClassif, CheckNorm, norm_batch, UnfreezeFcCrit, \
                SwitchAttribute, distrib_regul_regression, hist_lab, plot_results

