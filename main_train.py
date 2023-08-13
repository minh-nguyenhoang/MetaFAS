import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import Adam


from src.core.trainer import Trainer
from src.utils.data import ParallelInfiniteDataLoader
from src.dataset import StandardDataset, CombinedDataset
from src.core.config import Config
from src.core.model_config import ModelConfig
from src.network import Model
from src.utils.optimizer import SophiaG
from src.utils.callbacks import ModelCheckpoint
from src.utils.layers import Conv2d, LDConv2d, CDConv2d
from src.utils.transform import *
import torchvision.transforms as t

import random
import numpy as np

random.seed(Config.SEED)
np.random.seed(Config.SEED)


ModelConfig.IN_CHANNELS = 3
ModelConfig.META_SGD = False

Config.ACCUMULATED_GRADIENT_STEPS = 4


train_data_settings = {'get_real': None, 
                       'get_hsv': False if ModelConfig.IN_CHANNELS == 3 else True}

transform= t.Compose([RandomErasing(), Cutout(), RandomHorizontalFlip(), ToTensor(), RandomResizedCrop(), ColorJitter()])


model = Model(ModelConfig, conv_type = LDConv2d)
optim = Adam(model.parameters(), lr = 1e-4, weight_decay= 1e-4)
# ckpt_loss = ModelCheckpoint('weight_loss', 'test_loss', 'min', top_k= 2)
ckpt_acc = ModelCheckpoint('weight_acc_MAML_AGS4_ES', 'test_acc', 'max', top_k= 1)
ckpt_acc_epoch = ModelCheckpoint('weight_acc_epoch_MAML_AGS4_ES', 'epoch_acc', 'max', top_k= 1)

Config.NUMS_STEP = 20000

Config.BATCH_SIZE = 6

# trainer = Trainer(Config, model, optim, callbacks= {'val_loss': ckpt_loss, 'val_acc': ckpt_acc})
trainer = Trainer(Config, model, optim, callbacks= {'val_acc': ckpt_acc, 'val_acc_epoch': ckpt_acc_epoch}, use_logger = True)

# ## Casia_FASD
ds1 = CombinedDataset(['data/train_img', 'data/test_img'], **train_data_settings, transform = transform)
smplr1 = WeightedRandomSampler(ds1.get_weight(), len(ds1))
dl1 = DataLoader(ds1, batch_size= Config.BATCH_SIZE, sampler = smplr1)
## NUAA
ds2 = CombinedDataset(['data/NUAA/train', 'data/NUAA/test'], **train_data_settings, transform = transform)
smplr2 = WeightedRandomSampler(ds2.get_weight(), len(ds2))
dl2 = DataLoader(ds2, batch_size= Config.BATCH_SIZE, sampler = smplr2)
## Zalo
ds3 = StandardDataset('data/zalo_data', transform = t.Compose([ToTensor()]), **train_data_settings)

# smplr3 = WeightedRandomSampler(ds3.get_weight(), len(ds3))
# dl3 = DataLoader(ds3, batch_size= Config.BATCH_SIZE, sampler = smplr3)

weight = np.array(ds3.get_weight())
weight /= weight.sum()
indices_list = np.random.choice(list(range(len(ds3))), Config.BATCH_SIZE * 100, False,weight)
ds3 = torch.utils.data.Subset(ds3, indices_list)
dl3 = DataLoader(ds3, batch_size= Config.BATCH_SIZE)

## LCC_FASD
ds4 = CombinedDataset(['data/LCC_FASD_evaluation', 'data/LCC_FASD_training'], **train_data_settings, transform = transform)
smplr4 = WeightedRandomSampler(ds4.get_weight(), len(ds4))
dl4 = DataLoader(ds4, batch_size= Config.BATCH_SIZE, sampler = smplr4)





# dsv = StandardDataset('data/LCC_FASD_evaluation', get_real = None)
# smplrv = WeightedRandomSampler(dsv.get_weight(), len(dsv))
# dlv = DataLoader(dsv, batch_size= Config.BATCH_SIZE, sampler = smplrv)



trainer.fit( dl1,dl2, dl4, val_loader= dl3)

# ds2 = StandardDataset('data/train_img', **train_data_settings)
# smplr2 = WeightedRandomSampler(ds2.get_weight(), len(ds2))
# dl2 = DataLoader(ds2, batch_size= Config.BATCH_SIZE, sampler = smplr2)

# ds3 = StandardDataset('data/test_img', get_real = None, transform = t.Compose([ToTensor()]))
# smplr3 = WeightedRandomSampler(ds3.get_weight(), len(ds3))
# dl3 = DataLoader(ds3, batch_size= Config.BATCH_SIZE, sampler = smplr3)

# trainer.fit(dl2, val_loader= dl3)



