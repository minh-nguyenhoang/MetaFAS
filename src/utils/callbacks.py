from typing import Any
import torch
from copy import deepcopy
import os
from .misc import LimitedSizeList




class ModelCheckpoint:
    
    format = {'model_checkpoint': None, 
              'optimizer': None,
              'epoch': None,
              'value': None}
    name_format = 'model_epoch={}_{}={}_{}.pth'
    def __init__(self,root_dir: str = 'weights', criterion_name: str = 'val_loss', mode: str = 'min', top_k: int = 1, save_last: bool = True) -> None:
        '''
        Model checkpoint callback.

        Args:
            root_dir: str
                Name of the directory to save checkpoint. Default 'weights'
            criterion_name: str
                Name of the criterion used to evaluate.
            mode: str 
                Determine what to save (min for loss, max for accuracy, F1-score, etc.). Default: 'min'.
            top_k: int
                Determine how many best versions to save. Default: 1.\n
                Set this to 0 mean we do not save any version of best model checkpoint.
            save_last: bool
                Determine wether to save last model checkpoint on call

        Class properties:
            format: Optional(Dict, None)
                Decide the format of the checkpoint files. If 'None' only model checkpoint is saved. Default:
                    {'model_checkpoint': None, 
                     'optimizer': None,
                     'epoch': None,
                     'value': None}
            name_format: str
                Decide the format of the file name. Not yet support custom 'name_format'. Default:
                'model_epoch={epoch_num}_value={value_num}_{best/last}.pth'    

        '''
        assert isinstance(root_dir, str) and isinstance(criterion_name, str) and isinstance(mode, str) and isinstance(top_k, int) and isinstance(save_last, bool), \
                'Check the input type again!'
        assert mode in ['min', 'max'], "'mode' must be 'min' or 'max'!"
        assert top_k >=0, "'top_k' must be greater or equal 0!"
        assert top_k > 0 or save_last, "You must save atleast something!"

        if not os.path.exists(root_dir):
            os.makedirs(root_dir, exist_ok = True)

        self.root = root_dir
        self.criterion_name = criterion_name
        self.mode = mode
        self.top_k = top_k
        self.save_last = save_last
        self.criterion_value = float('inf')
        if top_k > 0:
            self.buffer = LimitedSizeList(top_k)
            self.score_buffer = LimitedSizeList(top_k, 'insert', 0. if mode == 'max' else float('inf'))

    def __call__(self, model, current_criterion_value, epoch, optimizer= None,) -> None:
        
        assert not(ModelCheckpoint.format is None and self.save_last is False), "Can't save any model if 'format' is 'None' and save_last' is 'False'"
        self.clear_previous_checkpoint()
        if ModelCheckpoint.format is None:
            format = model.state_dict()
        else:
            save_format = [model.state_dict(), optimizer.state_dict() if optimizer is not None else None, epoch, current_criterion_value]
            format = ModelCheckpoint.format.copy()
            for k in format.keys():
                format[k] = save_format.pop(0)

        if self.save_last:
            path = os.path.join(self.root,ModelCheckpoint.name_format.format(epoch, self.criterion_name, round(current_criterion_value,4), 'last'))
            torch.save(format, path)
        if self.top_k > 0 and ModelCheckpoint.format is not None:
            
            idx = self.get_index_by_score(current_criterion_value, self.score_buffer)
            self.buffer[idx] = format
            self.score_buffer[idx] = current_criterion_value
            k = 1
            for format_ in self.buffer:
                if format_ is not None:
                    path = os.path.join(self.root, ModelCheckpoint.name_format.format(format_['epoch'], self.criterion_name, round(format_['value'],4), f'best_k={k}'))
                    torch.save(format_, path)
                    k += 1
                
        
    def clear_format():
        ModelCheckpoint.format = None
    
    def reset_format():
        ModelCheckpoint.format = {'model_checkpoint': None, 
                                  'optimizer': None,
                                  'epoch': None,
                                  'value': None}
        
    def get_index_by_score(self, current_score, score_list):
        '''
        Return index that the value of its index is larger/smaller by 'mode'.
        '''
        idx = 0
        for score in score_list:
            if (current_score > score if self.mode == 'max' else current_score < score):
                return idx
            else:
                idx += 1
        return idx


    def clear_previous_checkpoint(self):
        # regex_model_name = ModelCheckpoint.name_format.format('.+?','.+?',f'best_k=.+?')
        dir_list = os.listdir(self.root)
        # dir_list_found = regex.findall(deepcopy(regex_model_name), ' '.join(dir_ for dir_ in dir_list))
        # print(dir_list_found)
        for dir_ in dir_list:
            os.remove(os.path.join(self.root,dir_)) 