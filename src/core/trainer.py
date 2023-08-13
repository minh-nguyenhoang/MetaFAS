import torch
from .config import Config
from tensorboardX import SummaryWriter
from ..utils.misc import DummyLogger, clip_grad_norm_
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from ..utils.data import ParallelInfiniteDataLoader, InfiniteDataLoader
from ..utils.meters import AverageMeter, ExponentialMeter
from ..utils.callbacks import ModelCheckpoint
import lightning as L
from typing import Dict
import random
import numpy as np

class Trainer:
    def __init__(self,config: Config, model, optimizer, scheduler = None, callbacks: dict| None = None, use_logger = True) -> None:
        self.cfg = config
        torch.backends.cudnn.benchmark = config.CUDNN_BENCHMARK
        self.scaler = torch.cuda.amp.GradScaler(init_scale=4096.0, enabled= config.ENABLE_GRADSCALER)
        self.fabric = L.Fabric(accelerator="auto", devices=1, strategy="auto")
        self.fabric.launch()
        self.fabric.seed_everything(config.SEED)
        random.seed(config.SEED)
        np.random.seed(config.SEED)

        self.model = self.fabric.setup_module(model)
        self.optimizer = self.fabric.setup_optimizers(optimizer)
        self.schduler = self.fabric.setup_optimizers(scheduler) if scheduler is not None else scheduler
        self.callbacks = callbacks
        if use_logger:
            self.logger = SummaryWriter(self.cfg.LOG_DIR)
        else:
            self.logger = DummyLogger(self.cfg.LOG_DIR)
        self.train_meter = self.get_meters('mtrain_loss','mtest_loss', 'acc')
        self.val_meter = self.get_meters('val_loss', 'val_acc', 'val_acc_epoch')
        

    def fit(self, *train_loader: DataLoader, val_loader: InfiniteDataLoader = None):
        self.model.train()
        
        train_loader = self.fabric.setup_dataloaders(*train_loader)
        if len(train_loader)== 1:
            train_loader = InfiniteDataLoader(*train_loader, step= Config.NUMS_STEP)
        else:
            train_loader = ParallelInfiniteDataLoader(*train_loader, step= Config.NUMS_STEP)            
        if val_loader is not None:
            val_loader = self.fabric.setup_dataloaders(val_loader)
            val_loader = InfiniteDataLoader(val_loader, step= -1)
        

        USE_AMP = self.cfg.USE_AMP
        tloss = 0
        # self.optimizer.zero_grad()
        for step, batch in (ep_bar := tqdm(train_loader, desc= "Training", leave= True, initial= train_loader.init_step)):
            output_ = self.model.training_step(step, batch, USE_AMP, self.train_meter['mtrain_loss'], self.train_meter['mtest_loss'], self.train_meter['acc'])

            loss = output_['loss']

            self.fabric.backward(self.scaler.scale(loss) / self.cfg.ACCUMULATED_GRADIENT_STEPS)

            if step % self.cfg.ACCUMULATED_GRADIENT_STEPS == 0:
                
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), 4.)

                self.scaler.step(self.optimizer)
                self.optimizer.zero_grad()
                self.scaler.update()
                tloss = 0
                ## update hessian of optimizer
#                 if hasattr(self.optimizer, 'update_hessian') and step % (10 * self.cfg.ACCUMULATED_GRADIENT_STEPS) == 0:
#                     self.optimizer.zero_grad()
#                     with torch.cuda.amp.autocast(enabled= USE_AMP):
#                         logits = self.model(batch[0])
#                     samp_dist = torch.distributions.Categorical(logits=logits)
#                     y_sample = samp_dist.sample()
#                     loss_sampled = self.model.criterion_cls(logits, y_sample)
#                     self.fabric.backward(self.scaler.scale(loss_sampled))
#                     self.optimizer.update_hessian()
                
            tmp_dict = self.train_meter.copy()
            # tmp_dict.update(self.val_meter)

            ep_bar.set_postfix({ k: v.avg for k,v in self.train_meter.items()})

            for k,v in output_.items():
#                 print(k,':', v.item())
                self.logger.add_scalar(k, v.item(), global_step= step)

            if step % self.cfg.VAL_EPOCH_EVERY_N_TRAIN_STEP == 0:
                print('>>>> Current LR: ', self.optimizer.optimizer.param_groups[0]['lr'])
                self.validation_epoch(val_loader, step)  ## Disable validation for meta-learning
#                 if self.callbacks is not None:
#                     for k, cb in self.callbacks.items():
#                         cb(self.model, self.train_meter[k].avg,step)




    @torch.no_grad()
    def validation_epoch(self, val_loader, training_step, step = 100):
        output_ =[]
        self.model.eval()
        for _ in (vpbar := tqdm(range(step), desc= 'Validating', leave= False)):
            step_, batch = next(val_loader)
            out = self.model.validating_step(step_, batch,self.cfg.USE_AMP, self.val_meter['val_loss'], self.val_meter['val_acc'])
            output_.append(out)
            vpbar.set_postfix({k: v.avg for k,v in self.val_meter.items()})
            
        label_pr = torch.cat([out['label_pr'] for out in output_], dim = 0)
        label_gt = torch.cat([out['label_gt'] for out in output_], dim = 0)
        epoch_val_acc = self.model.acc_metric(label_pr, label_gt)
        self.val_meter['val_acc_epoch'].update(epoch_val_acc.item())
        # self.logger.add_scalar('val_epoch_acc', epoch_val_acc.item(), global_step = training_step // self.cfg.VAL_EPOCH_EVERY_N_TRAIN_STEP )
        if self.callbacks is not None:
            for k, cb in self.callbacks.items():
                cb(self.model, self.val_meter[k].avg, training_step)


    def get_meters(self, *names):
        meters = {}
        for name in names:
            # meters[name] = AverageMeter(self.logger, name)
            if name == 'val_acc_epoch':
                meters[name] = ExponentialMeter(self.logger, name, weight = 1)
            else:
                meters[name] = ExponentialMeter(self.logger, name)
            
        
        return meters

    @torch.no_grad()
    def test(self, test_loader):
        test_loader = self.fabric.setup_dataloaders(test_loader)
