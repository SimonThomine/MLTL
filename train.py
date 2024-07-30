import os
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from dataset.mvtec import MVTecDefectDataset,MVTecTest
from utils.util import  AverageMeter,readYamlConfig,loadWeights,genLoaders,isBetterModel, set_seed
from models.extractor import teacherTimm
from models.MultiLayerTF import MLTF

class MltlTraining:          
    def __init__(self, data,device): 
        self.device = device
        self.validation_ratio = 0.2
        self.data_path = data['data_path']
        self.obj = data['obj']
        self.img_resize = data['TrainingData']['img_size']
        self.img_cropsize = data['TrainingData']['crop_size']
        self.num_epochs = data['TrainingData']['epochs']
        self.lr = data['TrainingData']['lr']
        self.batch_size = data['TrainingData']['batch_size'] 
        self.model_dir = data['save_path'] + "/models" + "/" + self.obj
        os.makedirs(self.model_dir, exist_ok=True)
        self.modelName = data['backbone']
        self.outIndices = data['out_indice']
        
        self.embedDim=data['embed_dim']
                    
        set_seed(42)    
        
        self.load_model()
        self.load_dataset()

        self.criterion = torch.nn.BCELoss()
        
        self.optimizer = torch.optim.Adam(self.classi.parameters(), lr=self.lr ,weight_decay=1e-5) 


    def load_dataset(self):
        kwargs = ({"num_workers": 8, "pin_memory": True} if torch.cuda.is_available() else {})
        #! Load dataset and create dataloaders
        train_dataset = MVTecDefectDataset(self.data_path,class_name=self.obj,resize=self.img_resize,cropsize=self.img_cropsize)
        self.train_loader,self.val_loader=genLoaders(train_dataset,self.validation_ratio,self.batch_size,**kwargs)
        

    def load_model(self):
        print("multi layer fine tuning " + self.modelName+" on "+self.obj)
        self.extractor = teacherTimm(backbone_name=self.modelName,out_indices=self.outIndices).to(self.device)
        filters,widths=self.extractor.simulate_output_sizes(self.img_cropsize,self.img_cropsize)
        self.classi=MLTF(filters=filters,widths=widths,embeddingDim=self.embedDim).to(self.device)
        
    def train(self):
        
        self.classi.train()
                
        best_loss = None
        epoch_bar = tqdm(total=len(self.train_loader) * self.num_epochs, desc="Training",unit="batch")
        
        for _ in range(1, self.num_epochs + 1):

            for data ,labels ,_ in self.train_loader:
                data = data.to(self.device)
                labels = labels.to(self.device).float()
                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                              
                    prediction = self.infer(data)
                    loss=self.criterion(prediction,labels)
                    loss.backward()
                    
                    self.optimizer.step()
                    
                    
                epoch_bar.set_postfix({"Loss": loss.item()})
                epoch_bar.update()

            val_loss = self.val()
                
            if (isBetterModel(best_loss, val_loss)):
                best_loss=val_loss
                self.save_checkpoint()

        epoch_bar.close()
        print("Training end.")

    def val(self):
        self.classi.eval()
        
        losses = AverageMeter()
        for data ,labels ,_ in self.val_loader:
            data = data.to(self.device)
            labels = labels.to(self.device).float()
            with torch.set_grad_enabled(False):
                
                prediction = self.infer(data)    
                loss=self.criterion(prediction,labels)
                
                losses.update(loss.item(), data.size(0))

        return losses.avg

    def save_checkpoint(self):
        state = {"model": self.classi.state_dict()}
        torch.save(state, os.path.join(self.model_dir, "classi.pth"))
        print("model saved")

    def test(self):
        
        self.classi=loadWeights(self.classi,self.model_dir,"classi.pth")

        kwargs = ({"num_workers": 4, "pin_memory": True} if torch.cuda.is_available() else {})
        test_dataset = MVTecTest(self.data_path,class_name=self.obj,is_train=False,resize=self.img_resize,cropsize=self.img_cropsize)
        batch_size_test = 1
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False, **kwargs)

        scores=[]
        gt_list = []
        progressBar = tqdm(test_loader) 

        for data, label in test_loader:
            gt_list.extend(label.cpu().numpy())
            label = label.to(self.device).float()
            data = data.to(self.device)
            with torch.set_grad_enabled(False):
                prediction=self.infer(data)
                progressBar.update()
              
            scores.extend([prediction.item()] if batch_size_test == 1 else prediction.tolist())

        
        progressBar.close()
        gt_list = np.asarray(gt_list)
        
        img_roc_auc = roc_auc_score(gt_list, scores)
        print(self.obj + " image ROCAUC memBank: %.3f" % (img_roc_auc))
        
        return img_roc_auc
    
    def infer(self, data):
        
        features=self.extractor(data)
        prediction=self.classi(features)
        
        return prediction
    
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data=readYamlConfig("config.yaml")
    distill = MltlTraining(data,device)
     
    if data['phase'] == "train":
        distill.train()
        distill.test()
    elif data['phase'] == "test":
        distill.test()
    else:
        print("Phase argument must be train or test.")
