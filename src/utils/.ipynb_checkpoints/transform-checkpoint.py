import numpy as np
import random
import math
import torch
import cv2
import torchvision.transforms.functional as vtf


def get_hsv_from_np(np_img, old_hsv: np.ndarray = None):
    ''' The old hsv image is to taking some statistic from it.
    Maybe rewrite the dataset to get the norm in the transform (BS) or apply the normalization at the end of the dataset (not really good either)
    '''

    if np.all(np_img < 1+1e-4):
        np_img = np.clip(np_img * 255, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(np_img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(float)

    if old_hsv is not None:
        if abs(old_hsv.min()) < 1e-6 and  abs(old_hsv.max() - 1) < 1e-6:
            # This (hopefully) is a 0-1 range normalization
            hmin = min(hsv.min(), 0)
            hmax = hsv.max()
            hsv = (hsv - hmin) / (hmax - hmin + 1e-9)
        else:
            # Assume this is a mean std normalization
            mean = old_hsv.mean((0,1))
            std = old_hsv.std((0,1))

            curr_mean = np.broadcast_to(hsv.mean((0,1)), hsv.shape)
            curr_std = np.broadcast_to(hsv.std((0,1)), hsv.shape)

            hsv_normal = (hsv - curr_mean) / ( curr_std + 1e-9)  #hsv have mean 0 and std 1

            hsv = hsv_normal * np.broadcast_to(std, hsv.shape) + np.broadcast_to(mean, hsv.shape)

    return hsv



class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, probability = 0.5, sl = 0.01, sh = 0.05, r1 = 0.5, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, sample):
        if len(sample) == 2:
            img, map_x = sample
        else:
            img, hsv, map_x = sample
        
        if random.uniform(0, 1) < self.probability:
            attempts = np.random.randint(1, 3)
            for attempt in range(attempts):
                area = img.shape[0] * img.shape[1]
           
                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.r1, 1/self.r1)
    
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
    
                if w < img.shape[1] and h < img.shape[0]:
                    x1 = random.randint(0, img.shape[0] - h)
                    y1 = random.randint(0, img.shape[1] - w)

                    img[x1:x1+h, y1:y1+w, 0] = self.mean[0]
                    img[x1:x1+h, y1:y1+w, 1] = self.mean[1]
                    img[x1:x1+h, y1:y1+w, 2] = self.mean[2]
                    

         
        if len(sample) == 3:
            return img, hsv, map_x
        return img, map_x



# Numpy array
class Cutout(object):
    def __init__(self, length=50):
        self.length = length

    def __call__(self, sample):
        if len(sample) == 2:
            img, map_x = sample
        else:
            img, hsv, map_x = sample
        h, w = img.shape[0], img.shape[1]    # Tensor [1][2],  nparray [0][1]
        mask = np.ones((h, w), np.uint8)
        y = np.random.randint(h)
        x = np.random.randint(w)
        length_new = np.random.randint(1, self.length)
        
        y1 = np.clip(y - length_new // 2, 0, h)
        y2 = np.clip(y + length_new // 2, 0, h)
        x1 = np.clip(x - length_new // 2, 0, w)
        x2 = np.clip(x + length_new // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = mask [:,:,None]
        
        img *= np.repeat(mask,img.shape[-1], -1)
        if len(sample) == 3:
            hsv *= np.repeat(mask,hsv.shape[-1], -1)
        map_x *= np.resize(mask, map_x.shape)
         
        if len(sample) == 3:
            return img, hsv, map_x
        return img, map_x



class RandomHorizontalFlip(object):
    """Horizontally flip the given Image randomly with a probability of p."""
    def __init__(self, p= .5) -> None:
        self.p = p
    def __call__(self, sample):
        if len(sample) == 2:
            img, map_x = sample
        else:
            img, hsv, map_x = sample
        
        # new_image_x = np.zeros(image_x.shape)
        # new_map_x = np.zeros(map_x.shape)

        p = random.random()
        if p < self.p:
        
            img = cv2.flip(img, 1)
            if len(sample) == 3:
                hsv = cv2.flip(hsv, 1)
            map_x = cv2.flip(map_x, 1)

        
        if len(sample) == 3:
            return img, hsv, map_x
        return img, map_x


class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample, im_size = (256,256)):
        if len(sample) == 2:
            img, map_x = sample
        else:
            img, hsv, map_x = sample
        
        # swap color axis because
        # numpy image: (batch_size) x H x W x C
        # torch image: (batch_size) x C X H X W
        img = torch.tensor(img)
        if len(sample) == 3:
            hsv = torch.tensor(hsv)
        map_x = torch.tensor(map_x)

        if img.shape[-len(im_size):] != torch.tensor(im_size):
            img = img.permute(2,0,1)
            if len(sample) == 3:
                hsv = hsv.permute(2,0,1)
        
        if len(sample) == 3:
            return img, hsv, map_x
        return img, map_x

# Tensor
class RandomResizedCrop(object):
    '''
    Class that performs Random Resized Crop to the input size.
    -------------------------------------------------------------------------------------
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, sl = 0.6, sh = 1., r1 = 0.7):
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, sample):
        if len(sample) == 2:
            img, map_x = sample
        else:
            img, hsv, map_x = sample
        c, ih, iw = img.shape[-3:]
        mh, mw = map_x.shape[-2:]
        area = img.shape[1] * img.shape[2]
    
        target_area = random.uniform(self.sl, self.sh) * area
        aspect_ratio = random.uniform(self.r1, 1/self.r1)

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))

        
        if h < img.shape[1] and w < img.shape[2]:
            x1 = random.randint(0, img.shape[1] - h)
            y1 = random.randint(0, img.shape[2] - w)
            x2 = x1 + h
            y2 = y1 + w
        
            img = img[:,x1:x2, y1:y2]
            img = vtf.resize(img, (ih, iw))

            if len(sample) == 3:
                hsv = hsv[:,x1:x2, y1:y2]
                hsv = vtf.resize(hsv, (ih, iw))

            h_ratio = ih/mh
            w_ratio = iw/mw
            x1 = int(x1 / h_ratio)
            x2 = int(x2 / h_ratio)
            y1 = int(y1 / w_ratio)
            y2 = int(y2 / w_ratio)

            map_x = map_x[x1:x2, y1:y2]
            
            map_x = vtf.resize(map_x.unsqueeze(0), (mh, mw)).squeeze(0)
        if len(sample) == 3:
            return img, hsv, map_x
        return img, map_x
    

class ColorJitter(object):
    def _check_input(self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True):
        import numbers
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(f"If {name} is a single number, it must be non negative.")
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError(f"{name} values should be between {bound}")
        else:
            raise TypeError(f"{name} should be a single number or a list/tuple with length 2.")

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value
    
    @staticmethod
    def get_params(
        brightness,
        contrast,
        saturation,
        hue,
    ):
        """Get the parameters for the randomized transform to be applied on image.

        Args:
            brightness (tuple of float (min, max), optional): The range from which the brightness_factor is chosen
                uniformly. Pass None to turn off the transformation.
            contrast (tuple of float (min, max), optional): The range from which the contrast_factor is chosen
                uniformly. Pass None to turn off the transformation.
            saturation (tuple of float (min, max), optional): The range from which the saturation_factor is chosen
                uniformly. Pass None to turn off the transformation.
            hue (tuple of float (min, max), optional): The range from which the hue_factor is chosen uniformly.
                Pass None to turn off the transformation.

        Returns:
            tuple: The parameters used to apply the randomized transform
            along with their random order.
        """
        fn_idx = torch.randperm(4)

        b = None if brightness is None else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
        c = None if contrast is None else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
        s = None if saturation is None else float(torch.empty(1).uniform_(saturation[0], saturation[1]))
        h = None if hue is None else float(torch.empty(1).uniform_(hue[0], hue[1]))

        return fn_idx, b, c, s, h
    
    def __init__(self, brightness = .3, contrast = .2, saturation = .1, hue = .1) -> None:
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)


    def __call__(self, sample):
        if len(sample) == 2:
            img, map_x = sample
        else:
            img, hsv, map_x = sample

        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue
        )

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = vtf.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = vtf.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = vtf.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = vtf.adjust_hue(img, hue_factor)

        if len(sample) == 3:
            np_img = img.permute(1,2,0).numpy()[:,:,::-1] #RGB2BGR
            hsv = get_hsv_from_np(np_img, hsv)
            
            hsv = vtf.to_tensor(hsv)
            return img, hsv, map_x
        return img, map_x