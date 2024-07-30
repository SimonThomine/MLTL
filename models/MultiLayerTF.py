import torch
from torch import nn
from torch.nn import functional as F

class MLTF(nn.Module):
    def __init__(self,filters=[],widths=[],embeddingDim=64):
        super(MLTF, self).__init__()
        
        self.embeddingDim=embeddingDim
        self.outputDim=max(widths)//2 
        
        
        in_chan=sum(filters)
        
        self.embedder=nn.Sequential(
            nn.Conv2d(in_chan, in_chan//2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_chan//2,momentum=0.01), 
            nn.Conv2d(in_chan//2, in_chan//4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_chan//4,momentum=0.01),
            nn.Conv2d(in_chan//4, embeddingDim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(embeddingDim,momentum=0.01),
            )
        
        self.upsample=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.pooling=nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)
        
        self.fc=nn.Linear(embeddingDim*self.outputDim*self.outputDim,1)
        self.sigmoid=nn.Sigmoid()
        
                              
    def forward(self, x):
        
        fuse=self.fuseLayers(x)
        embed=self.embedder(fuse)
        embed= self.pooling(embed)
        x=embed.view(-1,self.embeddingDim*self.outputDim*self.outputDim)
        
        x=self.fc(x)
        x=self.sigmoid(x).squeeze(1)
        return x

    def fuseLayers(self,x):
        xCat=x[0] 
        for layer in x[1:]:
            scaleFactor=xCat.shape[2]//layer.shape[2]
            upSampledLayer=F.interpolate(layer, scale_factor=scaleFactor, mode='bilinear', align_corners=True)
            xCat=torch.cat((xCat,upSampledLayer),dim=1)
        return xCat

