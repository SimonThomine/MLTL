import torch
from utils.util import readYamlConfig
from train import MltlTraining

textures=['wood','tile','grid','carpet','leather']
backbones=['efficientnet_b5']


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data=readYamlConfig("config.yaml")
    for backbone in backbones:
        data['backbone']=backbone
        for texture in textures:
            data['obj']=texture
            aurocMax=0
            trainerC = MltlTraining(data,device)
            trainerC.train()
            auroc=trainerC.test()