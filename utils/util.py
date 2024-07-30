import time
import random
import torch
import os
import yaml
import numpy as np
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def time_string():
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    string = '[{}]'.format(time.strftime(ISOTIMEFORMAT, time.localtime()))
    return string


def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    return need_hour, need_mins, need_secs


def itr_merge(*itrs):
    for itr in itrs:
        for v in itr:
            yield v
            
            
def readYamlConfig(configFileName):
    with open(configFileName) as f:
        data=yaml.safe_load(f)
        return data

def loadWeights(model,model_dir,alias):
    try:
        checkpoint = torch.load(os.path.join(model_dir, alias))
    except:
        raise Exception("Check saved model path.")
    model.load_state_dict(checkpoint["model"])
    model.eval() 
    for param in model.parameters():
        param.requires_grad = False
    return model


def genLoaders(train_dataset,validation_ratio, batch_size, **kwargs):
    img_nums = len(train_dataset)
    valid_num = int(img_nums * validation_ratio)
    train_num = img_nums - valid_num
    train_data, val_data = torch.utils.data.random_split(
        train_dataset, [train_num, valid_num]
    )
    return torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs),torch.utils.data.DataLoader(val_data, batch_size=8, shuffle=False, **kwargs)


def isBetterModel(best_loss, current_loss, threshold=0.01):
    if best_loss is None:
        return True
    elif current_loss < best_loss:
        return True
    
    
def set_seed(seed_value=42):
    """Fixe seed pour reproductibilité."""
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False