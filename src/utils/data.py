import torch

def get_infinite_data(data_loader, n, init_step :int = None, yield_step = True):
    step = 0 if init_step is None else init_step
    while True:
        for batch in data_loader:
            step = step + 1
            if n >= 0 and step > n:
                return
            if yield_step:
                yield step , batch
            else:
                yield batch


class InfiniteDataLoader:
    def __init__(self, dataloader, step, init_step = None, yield_step = True, *args, **kwargs) -> None:
        init_step = 0 if init_step is None else init_step
        self.len = None if step < 0 else (step - init_step) 
        self.init_step = init_step
        self.dataloader = get_infinite_data(dataloader, step, init_step, yield_step)
        
    def __len__(self):
        return max(self.len, 0)
    
    def __iter__(self):
        return self

    def __next__(self):
        return next(self.dataloader)
    

class ParallelInfiniteDataLoader:
    def __init__(self, *dataloaders, step = 20000, init_step = None, yield_step = True, yield_domain = True) -> None:
        self.yield_dmain = yield_domain
        self.yield_step = yield_step
        init_step = 0 if init_step is None else init_step
        self.init_step = init_step
        self.len = None if step < 0 else (step - init_step) 
        self.dataloaders = [get_infinite_data(dataloader, step, init_step, yield_step) for dataloader in dataloaders]
        
    def __len__(self):
        return max(self.len, 0)
    
    def __iter__(self):
        return self

    # def __next__(self):
    #     result = [next(dataloader) for dataloader in self.dataloaders]
    #     step = result[0][0]
    #     result = [ele for _, ele in result] ##[N,B,data]
    #     return step, *result

    def __next__(self):
        if self.yield_step:
            result = [next(dataloader) for dataloader in self.dataloaders]
            step = result[0][0]
            if self.yield_dmain:
                B = [res_[1][0].shape[0] for res_ in result]
                result = [(*ele,torch.tensor([enum] * B[enum])) for enum, (_, ele) in enumerate(result)] ##[N,data_size, B]
            else:
                result = [ele for (_, ele) in result] ##[N,data_size, B]
            result = list(zip(*result)) ##[data_size, N,B]
            result = tuple(torch.cat(data_ele, dim= 0) for data_ele in result) ##[data_size, N*B]
            return step, result
        else:
            result = [next(dataloader) for dataloader in self.dataloaders]   
            if self.yield_dmain:
                B = [res_[0].shape[0] for res_ in result]
                result = [(*ele,torch.tensor([enum] * B[enum])) for enum, ele in enumerate(result)]
            # else:
            #     result = [ele for (_, ele) in result] ##[N,data_size, B]
            result = list(zip(*result)) ##[data_size, N,B]
            result = tuple(torch.cat(data_ele, dim= 0) for data_ele in result) ##[data_size, N*B]
            return result