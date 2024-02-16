import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class MLP(nn.Module):
    
    def __init__(self, num_features, num_classes, depth=0, hdim=8, dropout=0.):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.depth = depth
        
        self.lin1 = torch.nn.Linear(self.num_features,  hdim)   
        self.linears = torch.nn.ModuleList([nn.Linear(hdim, hdim) for i in range(depth)])
        self.lin2 = torch.nn.Linear(hdim, self.num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, xin):
        x = F.leaky_relu(self.lin1(xin), 0.2)
        x = self.dropout(x)
        for i in range(self.depth): 
            x = F.leaky_relu(self.linears[i](x), 0.2) 
            x = self.dropout(x)
        res = F.leaky_relu(self.lin2(x), 0.2)
        return res

def _get_modified_resnet(resname, n_classes):
    resnet = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
    }[resname]()
    resnet.load_state_dict(torch.load(
        "./pretrained/resnet{}torchvision.pth".format(resname),
        map_location=torch.device('cpu')
    ))
    resnet.fc = nn.Linear(resnet.fc.in_features, n_classes)
    return resnet


def get_network(model_name, in_dim, n_classes, **kwargs):
    if model_name in ["rn18", "rn34", "rn50"]:
        return _get_modified_resnet(int(model_name[2:]), n_classes)
    elif model_name == 'mlp':
        return MLP(in_dim, n_classes, **kwargs)
    else:
        raise NotImplementedError("Unknown model {}".format(model_name))
