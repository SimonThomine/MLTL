import timm
import torch
import torch.nn as nn

class teacherTimm(nn.Module):
    def __init__(
        self,
        backbone_name="resnet18",
        out_indices=[1, 2, 3]
    ):
        super(teacherTimm, self).__init__()     
        self.feature_extractor = timm.create_model(
            backbone_name,
            pretrained=True,
            features_only=True,
            out_indices=out_indices 
        )
        
        self.feature_extractor.eval() 
        for param in self.feature_extractor.parameters():
            param.requires_grad = False   
        
    def forward(self, x):
        features_t = self.feature_extractor(x)
        return features_t
    
    def simulate_output_sizes(self, input_height, input_width):

        dummy_input = torch.randn(1, 3, input_height, input_width)
        

        dummy_input = dummy_input.to(next(self.feature_extractor.parameters()).device)
        
        with torch.no_grad():  
            output_features = self.feature_extractor(dummy_input)
    
        output_filters = [features.shape[1] for features in output_features]
        output_width = [features.shape[2] for features in output_features]
        
        return output_filters,output_width
    
    
