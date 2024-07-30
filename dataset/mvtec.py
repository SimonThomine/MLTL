import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from dataset.defectGenerator import DefectGenerator

class MVTecTest(Dataset):
    def __init__(self, dataset_path='../../datasets/MVTEC', class_name='carpet', is_train=True,
                 resize=224, cropsize=224):
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize

        self.x, self.y = self.loadDataset()

        self.transform_x = T.Compose([T.Resize(resize),T.CenterCrop(cropsize),T.ToTensor(),T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        self.transform_mask = T.Compose([T.Resize(resize, Image.NEAREST),T.CenterCrop(cropsize),T.ToTensor()])

    def __getitem__(self, idx):
        x, y= self.x[idx], self.y[idx]

        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        return x, y

    def __len__(self):
        return len(self.x)

    def loadDataset(self):
        
        x, y = [], []
        img_dir = os.path.join(self.dataset_path, self.class_name, 'test')
        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f) for f in os.listdir(img_type_dir)
                                     if (f.endswith('.png') or f.endswith('.jpg')or f.endswith('.tif'))])
            x.extend(img_fpath_list)

            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y)
    


class MVTecDefectDataset(Dataset):
    def __init__(self, dataset_path='../../datasets/MVTEC', class_name='carpet',resize=224, cropsize=224):
        
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.resize = resize
        self.cropsize = cropsize

        self.x = self.loadDataset()

        self.defectGenerator=DefectGenerator(resize_shape=[self.cropsize,self.cropsize])
    
        self.transform_x = T.Compose([T.Resize(resize),T.CenterCrop(cropsize),T.ToTensor()])
        
        self.normalize = T.Compose([T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
 
        
    def __getitem__(self, idx):
        
        x = self.x[idx]
        x = Image.open(x).convert('RGB')
        
        x = self.transform_x(x)
        x,isDefect,msk=self.defectGenerator.generateDefectRandomlyAndMsk(x)
        
        isDefect=torch.tensor(isDefect)
        
        x=self.normalize(x)
        
        
        return x,isDefect,msk

    def __len__(self):
        return len(self.x)

    def loadDataset(self):
        
        img_dir = os.path.join(self.dataset_path, self.class_name, 'train')
        img_type_dir = os.path.join(img_dir, 'good')
        x = sorted([os.path.join(img_type_dir, f) for f in os.listdir(img_type_dir) if (f.endswith('.png') or f.endswith('.jpg')or f.endswith('.tif'))])
        
        return list(x)
    