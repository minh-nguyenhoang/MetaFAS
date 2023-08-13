from .model.CDCN import FeatureExtractor, DepthEstor, Classifier
# from .model.ResNet18 import FeatureExtractor

from .utils.loss import KSubArcFace, contrast_depth_loss, MultiFocalLoss, HardTripletLoss, AsymmetricTripletLoss
from .core.model_config import ModelConfig
from .utils.layers import Conv2d
from .utils.misc import clip_norm
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torchmetrics.classification.accuracy import Accuracy

def map_dot2underscore(key):
    return key.replace('.','_')

def zero_param_grad(params):
    for p in params:
        if p.grad is not None:
            p.grad.zero_()

def clone_state_dict(model):
    clone_state_dict_ = {
        k: v.clone() for k,v in model.state_dict().items()
    }
    return clone_state_dict_

def define_task_lr_params(model, req_grad = True):
    task_lr = nn.ParameterDict()
    for key, val in model.named_parameters():
        # self.task_lr[key] = 1e-3 * torch.ones_like(val, requires_grad=True)
        key = map_dot2underscore(key)
        task_lr[key] = nn.Parameter(
            1e-3 * torch.ones_like(val, requires_grad=req_grad))
    return task_lr

class Model(nn.Module):
    def __init__(self, config:ModelConfig, conv_type = Conv2d, **kwargs) -> None:
        super().__init__()
        self.cfg = config
        self.featex = FeatureExtractor(in_channels= config.IN_CHANNELS, basic_conv = conv_type)

        self.depest = DepthEstor(basic_conv = conv_type)

        self.meta_leaner = Classifier(num_cls= 2, basic_conv = conv_type, **kwargs)

        
        self.task_lr = define_task_lr_params(self.meta_leaner, config.META_SGD)


        self.criterion_depth = lambda map_pr, map_gt: F.mse_loss(map_pr, map_gt) + contrast_depth_loss(map_pr, map_gt)
        self.criterion_cls = MultiFocalLoss(2, alpha= [0.25, 0.75])
        self.criterion_trplt = HardTripletLoss()

        self.criterion_atpl = AsymmetricTripletLoss(margin=1., hardest=False, angular = False, squared=False, asym_label= [1,2,3])
        
        self.acc_metric = Accuracy(num_classes= 2, task = 'multiclass')

    def forward(self, x):
        '''
        Label is used when training. For inferencing, disable the label for best performance
        '''
        feat_flt = self.featex(x)
        out_cls = self.meta_leaner(feat_flt, None)

        return out_cls
    
    def __train_forward(self,x, label = None, meta_params = None):
        feat_flt = self.featex(x)
        dep_pr = self.depest(feat_flt)
        out_cls = self.meta_leaner(feat_flt, label, meta_params)   

        return out_cls, dep_pr, feat_flt     
    
    def train1patch(self, batch_idx, batch, use_amp, loss_meter):
        img, map_gt, label_gt = batch
        with torch.cuda.amp.autocast(enabled= use_amp):
            label_pr, map_pr, feat = self.__train_forward(img, label_gt)
            
#             print(map_pr.shape, map_gt.shape)
            
            MSE_loss = self.criterion_depth(map_pr, map_gt)
            CLS_loss = self.criterion_cls(label_pr, label_gt)
            TPL_loss = self.criterion_trplt(feat, label_gt)
            loss =  MSE_loss + CLS_loss + TPL_loss

        loss_meter.update(loss.float().item(), step = batch_idx)

        return {'mse': MSE_loss,
                'cls': CLS_loss,
                'tpl': TPL_loss,
                'ttl': loss,
                'feat': feat}
    
    def test1patch(self, batch_idx, batch, use_amp, val_loss_meter, acc_meter, list_state_dict):
        img, map_gt, label_gt = batch
#         print(img.shape)

        # ori_state_dict = clone_state_dict(self.arcspoof)
        with torch.cuda.amp.autocast(enabled= use_amp):
            feat = self.featex(img)
            map_pr = self.depest(feat)
            
            MSE_loss = self.criterion_depth(map_pr, map_gt)
            TPL_loss = self.criterion_trplt(feat, label_gt)
            CLS_loss = 0
            ACC = 0

            for a_dict in list_state_dict:
                # self.arcspoof.load_state_dict(a_dict)
                label_pr = self.meta_leaner(feat, label_gt, a_dict)
                CLS_loss += self.criterion_cls(label_pr, label_gt)
                ACC += self.acc_metric(label_pr, label_gt.int())
                
            ACC = ACC / len(list_state_dict)
            
            loss =  MSE_loss + CLS_loss / len(list_state_dict) + TPL_loss

        val_loss_meter.update(loss.float().item(), step = batch_idx)
        acc_meter.update(ACC.float().item(), step = batch_idx)

        return {'mse': MSE_loss,
                'cls': CLS_loss,
                'tpl': TPL_loss,
                'ttl': loss,
                'acc': ACC,
                'feat': feat}



    def training_step(self, batch_idx, batch, use_amp, loss_meter, val_loss_meter, acc_meter):
        if len(batch) == 4:
            img, map_gt, label_gt, dmain = batch[0].float(), batch[1].float(), batch[2].float(), batch[3].float()

            dmain_list: list = dmain.unique().tolist()
            test_dmain_idx = random.choice(dmain_list)   ## The original RFMetaFAS also choose random test domain for each step?
            dmain_list.remove(test_dmain_idx)
            
            total_feat = []
            total_dmain_label = []


            ##################################
            ###       Meta-training        ###
            ##################################

            adapted_state_dicts = []

            Loss_CLS_train = 0
            Loss_MSE_train = 0
            Loss_TPL_train = 0

            for dmain_idx in dmain_list:
                data_idx = dmain == dmain_idx
                data_batch = (img[data_idx], map_gt[data_idx], label_gt[data_idx]) 
                output = self.train1patch(batch_idx, data_batch, use_amp, loss_meter)

                Loss_CLS_train += output['cls']
                Loss_MSE_train += output['mse']
                Loss_TPL_train += output['tpl']
                with torch.cuda.amp.autocast(enabled= use_amp):
                    feature = output['feat']
                    mask = 1 - label_gt[data_idx] # 0 for real face/ 1 for spoof face
                    pseudo_dmain_label = (dmain_idx +1) * torch.ones_like(mask) * mask #0 for real face/ dmain_idx for spoof face
                    # print(pseudo_dmain_label, label_gt[data_idx])
                    total_feat.append(feature)
                    total_dmain_label.append(pseudo_dmain_label)
                

#                 zero_param_grad(self.meta_leaner.parameters())

                grads_cls = torch.autograd.grad(output['cls'], self.meta_leaner.parameters(), create_graph= True)
                # grads_cls = clip_norm(grads_cls, 2.)

                fast_weights_cls = clone_state_dict(self.meta_leaner)
                
#                 for k,v in self.meta_leaner.named_parameters():
#                     fast_weights_cls[k] = v
        
                for (k,v), grad in zip(self.meta_leaner.named_parameters(), grads_cls):
                    fast_weights_cls[k] = (v - self.task_lr[map_dot2underscore(k)] * grad)


                adapted_state_dicts.append(fast_weights_cls)

            Loss_CLS_train = Loss_CLS_train / len(dmain_list)
            Loss_MSE_train = Loss_MSE_train / len(dmain_list)
            # Loss_TPL_train = Loss_TPL_train / len(dmain_list)

            ##################################
            ###       Meta-testing         ###
            ##################################

            data_idx = dmain == test_dmain_idx
            data_batch = (img[data_idx], map_gt[data_idx], label_gt[data_idx]) 
            output = self.test1patch(batch_idx, data_batch, use_amp, val_loss_meter, acc_meter, adapted_state_dicts)

            Loss_CLS_test = output['cls'] 
            Loss_MSE_test = output['mse'] 
            # Loss_TPL_test = output['tpl']

            ACC = output['acc']

            with torch.cuda.amp.autocast(enabled= use_amp):
                feature = output['feat']
                mask = 1 - label_gt[data_idx] # 0 for real face/ 1 for spoof face
                pseudo_dmain_label = (dmain_idx +1) * torch.ones_like(mask) * mask #0 for real face/ dmain_idx for spoof face
                # print(pseudo_dmain_label, label_gt[data_idx])
                total_feat.append(feature)
                total_dmain_label.append(pseudo_dmain_label)


            with torch.cuda.amp.autocast(enabled= use_amp):
                total_feat = torch.cat(total_feat, dim = 0)
                total_dmain_label = torch.cat(total_dmain_label, dim = 0)
            
                Loss_TPL = self.criterion_atpl(total_feat, total_dmain_label)

            
            ####################################
            ###           Update             ###
            ####################################

            loss_meta_train = Loss_CLS_train + Loss_MSE_train #+ Loss_TPL_train
            loss_meta_test = Loss_CLS_test + Loss_MSE_test #+ Loss_TPL_test

            loss_all = loss_meta_train + loss_meta_test #+ Loss_TPL

        return {'loss_meta_train': loss_meta_train,
                'loss_meta_test': loss_meta_test,
                'loss_cls_train': Loss_CLS_train,
                'loss_cls_test': Loss_CLS_test,
                'loss_mse_train': Loss_MSE_train,
                'loss_mse_test': Loss_MSE_test,
                'loss': loss_all,
                'acc': ACC}
    
    
    def validating_step(self, batch_idx, batch, use_amp, loss_meter, acc_meter):
        img, map_gt, label_gt = batch[0].float(), batch[1].float(), batch[2].float()

        with torch.cuda.amp.autocast(enabled= use_amp):
            label_pr, map_pr, feat = self.__train_forward(img)
            
#             print(map_pr.shape, map_gt.shape)
            
            MSE_loss = self.criterion_depth(map_pr, map_gt)
            CLS_loss = self.criterion_cls(label_pr, label_gt)
            TPL_loss = self.criterion_trplt(feat, label_gt)
            loss =  MSE_loss + CLS_loss #+ TPL_loss      

        loss_meter.update(loss.float().item(), step = batch_idx)
        acc = self.acc_metric(label_pr, label_gt)
        acc_meter.update(acc.float().item(), step = batch_idx)

        return {'mse': MSE_loss,
                'cls': CLS_loss,
                'tpl': TPL_loss,
                'acc': acc,
                'ttl': loss,
                'label_pr': label_pr,
                'label_gt': label_gt}



class TestModel(nn.Module):
    def __init__(self, config:ModelConfig, conv_type = Conv2d, **kwargs) -> None:
        super().__init__()
        self.cfg = config
        self.featex = FeatureExtractor(in_channels= config.IN_CHANNELS, basic_conv = conv_type)

        self.depest = DepthEstor(basic_conv = conv_type)

        self.meta_leaner = Classifier(num_cls= 2, basic_conv = conv_type, **kwargs)

        
        self.task_lr = define_task_lr_params(self.meta_leaner, config.META_SGD)



    def forward(self, x):
        '''
        Label is used when training. For inferencing, disable the label for best performance
        '''
        feat_flt = self.featex(x)
        out_cls = self.meta_leaner(feat_flt, None)

        return out_cls, feat_flt.flatten(start_dim = 1)


            

            


